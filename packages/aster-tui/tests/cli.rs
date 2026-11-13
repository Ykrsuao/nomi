// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::process::Command;

fn aster() -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_aster"));
    command.env_remove("ASTER_MODEL");
    command
}

#[test]
fn model_can_come_from_environment_or_explicit_argument() {
    for explicit in [false, true] {
        let mut command = aster();
        if explicit {
            command.args(["--model", "explicit-model"]);
        } else {
            command.env("ASTER_MODEL", "configured-model");
        }
        let output = command.output().unwrap();
        // Argument parsing succeeds, then the piped process stops before any backend starts.
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{stderr}");
        assert!(
            stderr.contains("TUI requires an interactive terminal"),
            "{stderr}"
        );
    }
}

#[test]
fn help_shows_the_configured_aster_model() {
    let output = aster()
        .env("ASTER_MODEL", "current-model")
        .arg("--help")
        .output()
        .unwrap();
    assert!(output.status.success());
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("ASTER_MODEL=current-model"), "{help}");
    assert!(!help.contains("--dev"), "{help}");
    assert!(!help.contains("--no-dev"), "{help}");
    assert!(!help.contains("ASTER_DEV"), "{help}");
}

#[test]
fn removed_developer_flags_are_rejected_before_starting_a_backend() {
    for flag in ["--dev", "--no-dev"] {
        let output = aster()
            .args(["--model", "test-model", flag])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{stderr}");
        assert!(stderr.contains("unexpected argument"), "{stderr}");
        assert!(stderr.contains(flag), "{stderr}");
        assert!(
            !stderr.contains("TUI requires an interactive terminal"),
            "{stderr}"
        );
    }
}

#[test]
fn legacy_developer_environment_does_not_change_startup() {
    for value in ["true", "false", "1", "0", "invalid"] {
        let output = aster()
            .env("ASTER_DEV", value)
            .args(["--model", "test-model"])
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(1), "{stderr}");
        assert!(
            stderr.contains("TUI requires an interactive terminal"),
            "{stderr}"
        );
    }
}
