// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseBoardUrl, parseCommand, stepTimeoutMs, upstreamPlan } from "./commands.ts";
import { wheelDirs } from "./wheels.ts";

test("only plain http addresses on the private IPv4 ranges are accepted", () => {
  assert.equal(parseBoardUrl("192.168.1.42"), "http://192.168.1.42");
  assert.equal(parseBoardUrl("http://10.0.0.7:8080/"), "http://10.0.0.7:8080");
  assert.equal(parseBoardUrl(" 172.20.3.4 "), "http://172.20.3.4");
  for (const bad of ["8.8.8.8", "127.0.0.1", "localhost", "https://192.168.1.2", "http://192.168.1.2/x", "http://u:p@192.168.1.2", "172.32.0.1", "192.168.1", "evil.com", ""])
    assert.equal(parseBoardUrl(bad), null, bad);
});

test("commands are parsed strictly and mapped to the firmware routes", () => {
  const q = (s: string) => parseCommand(new URLSearchParams(s));
  const plan = (s: string) => upstreamPlan(q(s)!);
  assert.deepEqual(plan("device=cam&cmd=capture").steps, ["/capture"]);
  assert.deepEqual(plan("device=cam&cmd=status").steps, ["/json"]);
  assert.deepEqual(plan("device=motors&cmd=move&motor=2&dir=1&speed=450&steps=200").steps, ["/move?motor=2&dir=1&speed=450&steps=200"]);
  assert.deepEqual(plan("device=motors&cmd=toggle_en&motor=1").steps, ["/toggle_en?motor=1"]);
  assert.deepEqual(plan("device=motors&cmd=distance").steps, ["/distance"]);
  assert.deepEqual(plan("device=drive&cmd=stop").steps, ["/stop?m=all"]);
  for (const bad of [
    "device=cam&cmd=restart",
    "device=motors&cmd=move&motor=3&dir=0&speed=450&steps=200",
    "device=motors&cmd=move&motor=1&dir=0&speed=450&steps=999999",
    "device=motors&cmd=move&motor=1&dir=2&speed=450&steps=10",
    "device=motors&cmd=move&motor=1&dir=0&speed=4.5&steps=10",
    "device=drive&cmd=run&dirs=1,1,1&speed=150&ms=500",
    "device=drive&cmd=run&dirs=1,1,2,1&speed=150&ms=500",
    "device=drive&cmd=run&dirs=0,0,0,0&speed=150&ms=500",
    "device=drive&cmd=run&dirs=1,1,1,1&speed=300&ms=500",
    "device=drive&cmd=run&dirs=1,1,1,1&speed=150&ms=50",
    "device=drive&cmd=jump",
    "device=x&cmd=capture",
  ])
    assert.equal(q(bad), null, bad);
});

test("a drive move sets each wheel, starts them, waits, and always ends with a stop", () => {
  const p = upstreamPlan(parseCommand(new URLSearchParams("device=drive&cmd=run&dirs=1,1,-1,0&speed=150&ms=800"))!);
  assert.deepEqual(p.steps, [
    "/stop?m=all",
    "/speed?m=1&val=150", "/speed?m=2&val=150", "/speed?m=3&val=150",
    "/dir?m=1&val=1", "/dir?m=2&val=1", "/dir?m=3&val=-1",
    "/start?m=1", "/start?m=2", "/start?m=3",
  ]);
  assert.equal(p.holdMs, 800);
  const all = upstreamPlan(parseCommand(new URLSearchParams("device=drive&cmd=run&dirs=1,1,1,1&speed=150&ms=800"))!);
  assert.deepEqual(all.steps, ["/stop?m=all", "/speed?m=all&val=150", "/dir?m=all&val=1", "/start?m=all"]);
  const turn = upstreamPlan(parseCommand(new URLSearchParams("device=drive&cmd=run&dirs=-1,-1,1,1&speed=150&ms=800"))!);
  assert.deepEqual(turn.steps, ["/stop?m=all", "/speed?m=all&val=150", "/dir?m=1&val=-1", "/dir?m=2&val=-1", "/dir?m=3&val=1", "/dir?m=4&val=1", "/start?m=all"]);
  assert.equal(p.always, "/stop?m=all");
});

test("wheel directions follow the side of each motor and its inversion", () => {
  const sides = ["left", "left", "right", "right"] as const;
  assert.deepEqual(wheelDirs("forward", sides, [false, false, false, false]), [1, 1, 1, 1]);
  assert.deepEqual(wheelDirs("back", sides, [false, false, false, false]), [-1, -1, -1, -1]);
  assert.deepEqual(wheelDirs("left", sides, [false, false, false, false]), [-1, -1, 1, 1]);
  assert.deepEqual(wheelDirs("right", sides, [false, false, true, false]), [1, 1, 1, -1]);
});

test("a long camera move gets a timeout long enough for the motor to finish", () => {
  const c = parseCommand(new URLSearchParams("device=motors&cmd=move&motor=1&dir=0&speed=100&steps=1000"))!;
  assert.ok(stepTimeoutMs(c) > 10_000);
});
