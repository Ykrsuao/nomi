// SPDX-FileCopyrightText: Copyright (c) 2026 Aster contributors
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import { build } from "esbuild";

const bundle = await build({
  entryPoints: [
    fileURLToPath(new URL("../src/api/traces.ts", import.meta.url)),
  ],
  bundle: true,
  format: "esm",
  platform: "node",
  write: false,
});
const {
  convertOtlpSpansToEvents,
  normalizeTraceAttributes,
  fetchTraceResource,
} = await import(
  `data:text/javascript;base64,${Buffer.from(bundle.outputFiles[0].text).toString("base64")}`
);

function attr(key, value) {
  return { key, value: { stringValue: value } };
}

function span(attributes, resource = []) {
  return {
    traceId: "trace-1",
    spanId: "span-1",
    name: "method.handle",
    kind: 1,
    startTimeUnixNano: "1000000",
    endTimeUnixNano: "2000000",
    attributes,
    status: { code: 1 },
    _resource: { attributes: resource },
  };
}

test("saved metadata restores plugin and prompt while preserving original trace values", () => {
  const prompt = "The user wrote nooa and nemo_oo_agents in their source code.";
  const input = [
    span([
      attr("nooa.viewer.plugin", "method"),
      attr("nemo_oo_agents.system_message", prompt),
      attr("nemo_oo_agents.version", "0.0.7"),
    ]),
  ];
  const before = structuredClone(input);
  const [event] = convertOtlpSpansToEvents(input);
  assert.equal(event.type, "span.method.handle");
  assert.equal(event.attributes["aster.system_message"], prompt);
  assert.equal(event.attributes["aster.version"], "0.0.7");
  assert.equal(
    Object.hasOwn(event.attributes, "nemo_oo_agents.system_message"),
    false,
  );
  assert.deepEqual(input, before);
});

test("current metadata wins namespace collisions independent of key order", () => {
  for (const entries of [
    [
      ["nemo_oo_agents.viewer.plugin", "oldest"],
      ["nooa.viewer.plugin", "old"],
      ["aster.viewer.plugin", "method"],
    ],
    [
      ["aster.viewer.plugin", "method"],
      ["nooa.viewer.plugin", "old"],
      ["nemo_oo_agents.viewer.plugin", "oldest"],
    ],
  ]) {
    const [event] = convertOtlpSpansToEvents([
      span(entries.map(([key, value]) => attr(key, value))),
    ]);
    assert.equal(event.type, "span.method.handle");
  }
});

test("span-local historical values override resource defaults after normalization", () => {
  const [event] = convertOtlpSpansToEvents([
    span(
      [attr("nemo_oo_agents.user_message", "local prompt")],
      [attr("aster.user_message", "resource default")],
    ),
  ]);
  assert.equal(event.attributes["aster.user_message"], "local prompt");
});

test("only reserved top-level keys are normalized; nested user objects stay intact", () => {
  const payload = {
    "nooa.user_field": "literal payload",
    text: "NVIDIA OO Agents",
  };
  const input = {
    payload,
    "service.name": "my-service",
    "nemo_oo_agents.user_message": "message",
  };
  const before = structuredClone(input);
  const normalized = normalizeTraceAttributes(input);
  assert.equal(normalized.payload, payload);
  assert.equal(normalized["service.name"], "my-service");
  assert.equal(normalized["aster.user_message"], "message");
  assert.deepEqual(input, before);
});

test("resource endpoint uses the same metadata normalization", async (t) => {
  const input = { "nemo_oo_agents.version": "0.0.7", batch_id: "batch-1" };
  t.mock.method(
    globalThis,
    "fetch",
    async () => new Response(JSON.stringify(input)),
  );
  assert.deepEqual(await fetchTraceResource("saved-session"), {
    "aster.version": "0.0.7",
    batch_id: "batch-1",
  });
  assert.deepEqual(input, {
    "nemo_oo_agents.version": "0.0.7",
    batch_id: "batch-1",
  });
});
