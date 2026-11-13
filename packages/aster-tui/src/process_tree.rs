// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Bind backend descendants to the transport before any backend code executes.

pub(crate) use platform::ProcessTree;

#[cfg(windows)]
mod platform {
    use std::io;
    use std::mem::size_of;
    use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
    use std::ptr::{null, null_mut};
    use std::time::Duration;

    use anyhow::{bail, Context, Result};
    use tokio::process::{Child, Command};
    use windows_sys::Win32::Foundation::{FILETIME, INVALID_HANDLE_VALUE};
    use windows_sys::Win32::System::Diagnostics::ToolHelp::{
        CreateToolhelp32Snapshot, Thread32First, Thread32Next, TH32CS_SNAPTHREAD, THREADENTRY32,
    };
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JobObjectBasicAccountingInformation,
        JobObjectExtendedLimitInformation, QueryInformationJobObject, SetInformationJobObject,
        TerminateJobObject, JOBOBJECT_BASIC_ACCOUNTING_INFORMATION,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    use windows_sys::Win32::System::Threading::{
        GetThreadTimes, OpenThread, ResumeThread, CREATE_NO_WINDOW, CREATE_SUSPENDED,
        THREAD_QUERY_LIMITED_INFORMATION, THREAD_SUSPEND_RESUME,
    };

    // OwnedHandle is Send and closes the non-inheritable job on every error/drop
    // path. The backend cannot keep the job alive by inheriting our handle.
    pub(crate) struct ProcessTree {
        job: OwnedHandle,
    }

    impl ProcessTree {
        fn new() -> Result<Self> {
            // SAFETY: null attributes create a non-inheritable, unnamed job.
            let handle = unsafe { CreateJobObjectW(null(), null()) };
            if handle.is_null() {
                return Err(io::Error::last_os_error()).context("Could not create backend job");
            }
            // SAFETY: successful CreateJobObjectW transferred this handle to us.
            let job = unsafe { OwnedHandle::from_raw_handle(handle) };
            let mut limits = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: the job is live and the buffer matches the information class.
            if unsafe {
                SetInformationJobObject(
                    job.as_raw_handle(),
                    JobObjectExtendedLimitInformation,
                    (&limits as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            } == 0
            {
                return Err(io::Error::last_os_error()).context("Could not configure backend job");
            }
            Ok(Self { job })
        }

        pub(crate) fn spawn(command: &mut Command) -> Result<(Child, Self)> {
            let tree = Self::new()?;
            // Assigning an already-running child permits it to spawn descendants
            // before assignment. Suspend its primary thread until containment is
            // established. Do not request breakaway: inherited jobs nest normally.
            command.creation_flags(CREATE_NO_WINDOW | CREATE_SUSPENDED);
            let child = command
                .spawn()
                .context("Could not start the Python ACP backend")?;
            tree.assign(&child)?;
            resume_primary_thread(child.id().context("Backend process ID is unavailable")?)?;
            Ok((child, tree))
        }

        fn assign(&self, child: &Child) -> Result<()> {
            let process = child
                .raw_handle()
                .context("Backend process handle is unavailable")?;
            // SAFETY: both handles remain owned/live throughout the call. An empty
            // job with no UI restrictions supports a process already in a parent job.
            if unsafe { AssignProcessToJobObject(self.job.as_raw_handle(), process) } == 0 {
                return Err(io::Error::last_os_error())
                    .context("Could not contain the backend process tree");
            }
            Ok(())
        }

        pub(crate) fn terminate(&mut self) -> Result<()> {
            // SAFETY: the owned job handle is live. This includes nested child jobs.
            if unsafe { TerminateJobObject(self.job.as_raw_handle(), 1) } == 0 {
                return Err(io::Error::last_os_error())
                    .context("Could not terminate the backend process tree");
            }
            Ok(())
        }

        pub(crate) async fn terminate_and_wait(&mut self) -> Result<()> {
            self.terminate()?;
            tokio::time::timeout(Duration::from_secs(5), async {
                loop {
                    let mut info = JOBOBJECT_BASIC_ACCOUNTING_INFORMATION::default();
                    // SAFETY: the live job writes into a correctly sized buffer.
                    if unsafe {
                        QueryInformationJobObject(
                            self.job.as_raw_handle(),
                            JobObjectBasicAccountingInformation,
                            (&mut info as *mut JOBOBJECT_BASIC_ACCOUNTING_INFORMATION).cast(),
                            size_of::<JOBOBJECT_BASIC_ACCOUNTING_INFORMATION>() as u32,
                            null_mut(),
                        )
                    } == 0
                    {
                        return Err(io::Error::last_os_error())
                            .context("Could not inspect backend cleanup");
                    }
                    if info.ActiveProcesses == 0 {
                        return Ok(());
                    }
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            })
            .await
            .context("Backend descendants did not exit after being terminated")?
        }
    }

    fn resume_primary_thread(pid: u32) -> Result<()> {
        // Tokio retains the process handle but not CreateProcess's primary-thread
        // handle. Locate its earliest-created thread using documented APIs rather
        // than assuming Toolhelp enumeration order (external software can inject
        // additional threads even before the primary thread is resumed).
        // SAFETY: snapshot creation needs no borrowed memory.
        let snapshot = unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) };
        if snapshot == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error()).context("Could not inspect suspended backend");
        }
        // SAFETY: snapshot creation succeeded and transfers ownership.
        let snapshot = unsafe { OwnedHandle::from_raw_handle(snapshot) };
        let mut entry = THREADENTRY32 {
            dwSize: size_of::<THREADENTRY32>() as u32,
            ..Default::default()
        };
        // SAFETY: the snapshot is live and the entry has the required size.
        let mut found = unsafe { Thread32First(snapshot.as_raw_handle(), &mut entry) };
        let mut primary: Option<(u64, OwnedHandle)> = None;
        while found != 0 {
            if entry.th32OwnerProcessID == pid {
                // SAFETY: only the suspended child's thread is opened; the process
                // handle prevents PID reuse while we inspect the snapshot.
                let thread = unsafe {
                    OpenThread(
                        THREAD_SUSPEND_RESUME | THREAD_QUERY_LIMITED_INFORMATION,
                        0,
                        entry.th32ThreadID,
                    )
                };
                if thread.is_null() {
                    return Err(io::Error::last_os_error())
                        .context("Could not open suspended backend thread");
                }
                // SAFETY: successful OpenThread transferred ownership.
                let thread = unsafe { OwnedHandle::from_raw_handle(thread) };
                let mut creation = FILETIME::default();
                let mut exit = FILETIME::default();
                let mut kernel = FILETIME::default();
                let mut user = FILETIME::default();
                // SAFETY: the live thread has query access and all outputs point
                // to valid FILETIME values for the duration of the call.
                if unsafe {
                    GetThreadTimes(
                        thread.as_raw_handle(),
                        &mut creation,
                        &mut exit,
                        &mut kernel,
                        &mut user,
                    )
                } == 0
                {
                    return Err(io::Error::last_os_error())
                        .context("Could not identify backend primary thread");
                }
                let created =
                    (u64::from(creation.dwHighDateTime) << 32) | u64::from(creation.dwLowDateTime);
                if primary
                    .as_ref()
                    .is_none_or(|(earliest, _)| created < *earliest)
                {
                    primary = Some((created, thread));
                }
            }
            entry.dwSize = size_of::<THREADENTRY32>() as u32;
            // SAFETY: same valid snapshot and entry as Thread32First.
            found = unsafe { Thread32Next(snapshot.as_raw_handle(), &mut entry) };
        }
        let (_, thread) =
            primary.context("Could not find the suspended backend's primary thread")?;
        // SAFETY: the earliest thread is the primary thread, initially suspended.
        let count = unsafe { ResumeThread(thread.as_raw_handle()) };
        if count == u32::MAX {
            return Err(io::Error::last_os_error()).context("Could not resume the backend");
        }
        if count != 1 {
            bail!("Backend primary thread had an unexpected suspend count: {count}");
        }
        Ok(())
    }
}

#[cfg(unix)]
mod platform {
    use std::io;

    use anyhow::{Context, Result};
    use tokio::process::{Child, Command};

    pub(crate) struct ProcessTree {
        group: i32,
        terminated: bool,
    }

    impl ProcessTree {
        pub(crate) fn spawn(command: &mut Command) -> Result<(Child, Self)> {
            // setpgid runs in the child before exec, so descendants inherit the
            // isolated group from their first instruction.
            command.process_group(0);
            let child = command
                .spawn()
                .context("Could not start the Python ACP backend")?;
            let group = child.id().context("Backend process ID is unavailable")? as i32;
            Ok((
                child,
                Self {
                    group,
                    terminated: false,
                },
            ))
        }

        pub(crate) fn terminate(&mut self) -> Result<()> {
            if !self.terminated {
                // SAFETY: the positive PID was obtained from our spawned child;
                // negating it targets only that child's dedicated process group.
                if unsafe { libc::kill(-self.group, libc::SIGKILL) } != 0 {
                    let error = io::Error::last_os_error();
                    if error.raw_os_error() != Some(libc::ESRCH) {
                        return Err(error).context("Could not terminate the backend process group");
                    }
                }
                self.terminated = true;
            }
            Ok(())
        }

        pub(crate) async fn terminate_and_wait(&mut self) -> Result<()> {
            // The transport reaps its direct child. Grandchildren are reaped by
            // their new parent; waiting for killpg(..., 0) would include zombies.
            self.terminate()
        }
    }

    impl Drop for ProcessTree {
        fn drop(&mut self) {
            let _ = self.terminate();
        }
    }
}
