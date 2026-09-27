// node --test (Node ≥ 22.18 strips the types): pnpm test:scripts
import { test } from "node:test";
import assert from "node:assert/strict";
import { parseBoardUrl, parseCommand, stepTimeoutMs, upstreamPlan } from "./commands.ts";
import { applyInvert, DEFAULT_WHEEL_MOVES, validMove, wheelInvertOr, wheelMovesOr } from "./wheels.ts";

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

test("wheel moves: saved ones are used when valid, the default otherwise", () => {
  const diagonal = { forward: [1, 1, 1, 1], back: [-1, -1, -1, -1], left: [1, 0, 0, -1], right: [-1, 0, 0, 1] };
  assert.deepEqual(wheelMovesOr(diagonal), diagonal);
  assert.deepEqual(wheelMovesOr({ ...diagonal, left: [0, 0, 0, 0] }).left, DEFAULT_WHEEL_MOVES.left);
  assert.deepEqual(wheelMovesOr({ ...diagonal, right: [2, 0, 0, 1] }).right, DEFAULT_WHEEL_MOVES.right);
  assert.deepEqual(wheelMovesOr(undefined), DEFAULT_WHEEL_MOVES);
  assert.equal(validMove([0, 0, 0, 0]), false);
  assert.equal(validMove([1, -1, 0, 0]), true);
});

test("a motor wired the other way round is reversed in every move", () => {
  assert.deepEqual(applyInvert([1, 1, 1, 1], [false, true, false, false]), [1, -1, 1, 1]);
  assert.deepEqual(applyInvert([-1, 0, 1, 1], [false, true, false, false]), [-1, 0, 1, 1]);
  assert.deepEqual(wheelInvertOr(undefined), [false, true, false, false]);
  assert.deepEqual(wheelInvertOr([true, false, false, false]), [true, false, false, false]);
  assert.deepEqual(wheelInvertOr([1, 2]), [false, true, false, false]);
});

test("a long camera move gets a timeout long enough for the motor to finish", () => {
  const c = parseCommand(new URLSearchParams("device=motors&cmd=move&motor=1&dir=0&speed=100&steps=1000"))!;
  assert.ok(stepTimeoutMs(c) > 10_000);
});

test("hold to drive: go sets and starts the wheels with no stop; keep sends nothing to the board", () => {
  const go = upstreamPlan(parseCommand(new URLSearchParams("device=drive&cmd=go&dirs=1,1,1,1&speed=150"))!);
  assert.deepEqual(go.steps, ["/stop?m=all", "/speed?m=all&val=150", "/dir?m=all&val=1", "/start?m=all"]);
  assert.equal(go.always, undefined);
  assert.equal(go.holdMs, undefined);
  assert.deepEqual(upstreamPlan(parseCommand(new URLSearchParams("device=drive&cmd=keep"))!).steps, []);
  assert.equal(parseCommand(new URLSearchParams("device=drive&cmd=go&dirs=1,1,1,1&speed=999")), null);
});
