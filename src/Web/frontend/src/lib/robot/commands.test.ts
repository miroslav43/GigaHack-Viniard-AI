// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseBoardUrl, parseCommand, timeoutMs, upstreamPath } from "./commands.ts";

test("only plain http addresses on the private IPv4 ranges are accepted", () => {
  assert.equal(parseBoardUrl("192.168.1.42"), "http://192.168.1.42");
  assert.equal(parseBoardUrl("http://10.0.0.7:8080/"), "http://10.0.0.7:8080");
  assert.equal(parseBoardUrl(" 172.20.3.4 "), "http://172.20.3.4");
  for (const bad of ["8.8.8.8", "127.0.0.1", "localhost", "https://192.168.1.2", "http://192.168.1.2/x", "http://u:p@192.168.1.2", "172.32.0.1", "192.168.1", "evil.com", ""])
    assert.equal(parseBoardUrl(bad), null, bad);
});

test("commands are parsed strictly and mapped to the firmware routes", () => {
  const q = (s: string) => parseCommand(new URLSearchParams(s));
  assert.equal(upstreamPath(q("device=cam&cmd=capture")!), "/capture");
  assert.equal(upstreamPath(q("device=cam&cmd=status")!), "/json");
  assert.equal(upstreamPath(q("device=motors&cmd=move&motor=2&dir=1&speed=450&steps=200")!), "/move?motor=2&dir=1&speed=450&steps=200");
  assert.equal(upstreamPath(q("device=motors&cmd=toggle_en&motor=1")!), "/toggle_en?motor=1");
  assert.equal(upstreamPath(q("device=drive&cmd=forward&ms=800")!), "/drive?cmd=forward&ms=800");
  for (const bad of [
    "device=cam&cmd=restart",
    "device=motors&cmd=move&motor=3&dir=0&speed=450&steps=200",
    "device=motors&cmd=move&motor=1&dir=0&speed=450&steps=999999",
    "device=motors&cmd=move&motor=1&dir=2&speed=450&steps=10",
    "device=motors&cmd=move&motor=1&dir=0&speed=4.5&steps=10",
    "device=drive&cmd=jump&ms=500",
    "device=drive&cmd=forward&ms=50",
    "device=x&cmd=capture",
  ])
    assert.equal(q(bad), null, bad);
});

test("a long move gets a timeout long enough for the motor to finish", () => {
  const c = parseCommand(new URLSearchParams("device=motors&cmd=move&motor=1&dir=0&speed=100&steps=1000"))!;
  assert.ok(timeoutMs(c) > 10_000);
});
