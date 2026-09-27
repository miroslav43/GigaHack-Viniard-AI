#!/usr/bin/env node
// A stand-in for the field robot's three ESP32 boards, to try /robot without the hardware. Same routes as the
// firmware: camera (/stream MJPEG, /capture, /flash/*, /json), camera motors + distance sensor (/move, /toggle_en,
// /distance) and the wheels' four DC motors (/dir, /speed, /start, /stop, /status).
// Every board listens on this computer's address, one port each; the page accepts only private IPv4 addresses, so
// use the computer's LAN address (printed at start), not localhost.
//   node scripts/robot/fake-robot.mjs [--host 192.168.1.20] [--port 8081]   (cam = port, motors = port+1, drive = port+2)
import { createServer } from "node:http";
import { networkInterfaces } from "node:os";
import { deflateSync } from "node:zlib";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? process.argv[i + 1] : fallback;
};
const lanAddress = () =>
  Object.values(networkInterfaces())
    .flat()
    .find((a) => a && a.family === "IPv4" && !a.internal && /^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(a.address))?.address;

const HOST = arg("host", lanAddress() ?? "127.0.0.1");
const PORT = Number(arg("port", "8081"));

// ---- a tiny picture that changes with the camera angle (a PNG served as the "JPEG": browsers sniff the bytes) ----
const crcTable = Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const b of buf) c = crcTable[(c ^ b) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
const chunk = (type, data) => {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
};
function picture(pan, tilt, flash) {
  const w = 160, h = 120;
  const raw = Buffer.alloc((w * 3 + 1) * h);
  for (let y = 0; y < h; y++) {
    raw[y * (w * 3 + 1)] = 0;
    for (let x = 0; x < w; x++) {
      const i = y * (w * 3 + 1) + 1 + x * 3;
      const rowStripe = Math.floor((x + pan) / 12) % 2 === 0; // vine rows that slide with the pan
      const sky = y < 40 - tilt / 3;
      const boost = flash ? 60 : 0;
      raw[i] = Math.min(255, (sky ? 150 : rowStripe ? 60 : 140) + boost);
      raw[i + 1] = Math.min(255, (sky ? 190 : rowStripe ? 140 : 110) + boost);
      raw[i + 2] = Math.min(255, (sky ? 235 : rowStripe ? 50 : 70) + boost);
    }
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8;
  ihdr[9] = 2;
  return Buffer.concat([Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]), chunk("IHDR", ihdr), chunk("IDAT", deflateSync(raw)), chunk("IEND", Buffer.alloc(0))]);
}

const state = {
  pan: 0,
  tilt: 0,
  flash: false,
  soft: { 1: true, 2: true },
  wheels: [1, 2, 3, 4].map(() => ({ dir: 1, speed: 150, running: false })),
};
const text = (res, status, body) => res.writeHead(status, { "Content-Type": "text/plain; charset=UTF-8" }).end(body);
const log = (board, what) => console.log(`[${board}] ${what}`);

function camera(req, res) {
  const url = new URL(req.url, "http://x");
  switch (url.pathname) {
    case "/stream": {
      res.writeHead(200, { "Content-Type": "multipart/x-mixed-replace;boundary=frame" });
      const timer = setInterval(() => {
        const img = picture(state.pan, state.tilt, state.flash);
        res.write(`\r\n--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ${img.length}\r\n\r\n`);
        res.write(img);
      }, 200);
      req.on("close", () => clearInterval(timer));
      return;
    }
    case "/capture":
      log("cam", `capture (pan ${state.pan}°, tilt ${state.tilt}°)`);
      return res.writeHead(200, { "Content-Type": "image/jpeg" }).end(picture(state.pan, state.tilt, state.flash));
    case "/flash/on":
    case "/flash/off":
    case "/flash/toggle":
      state.flash = url.pathname.endsWith("on") ? true : url.pathname.endsWith("off") ? false : !state.flash;
      log("cam", `flash ${state.flash ? "ON" : "OFF"}`);
      return text(res, 200, state.flash ? "FLASH ON" : "FLASH OFF");
    case "/json":
      return res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify({ device: "FAKE ESP32-CAM", ip: HOST, flash: state.flash ? "ON" : "OFF" }));
    default:
      return text(res, 404, `Ruta inexistenta: ${url.pathname}`);
  }
}

function motors(req, res) {
  const url = new URL(req.url, "http://x");
  const q = url.searchParams;
  if (url.pathname === "/move") {
    const m = Number(q.get("motor")), dir = Number(q.get("dir")), speed = Number(q.get("speed")), steps = Number(q.get("steps"));
    if (![1, 2].includes(m) || !speed || !steps) return text(res, 400, "Eroare: Parametri lipsă!");
    const deg = (steps * 360) / 200;
    if (m === 1) state.pan += dir === 0 ? deg : -deg;
    else state.tilt += dir === 0 ? deg : -deg;
    state.soft[m] = false;
    log("motors", `motor ${m} dir ${dir} ${steps} steps @ ${speed} pps → pan ${state.pan}°, tilt ${state.tilt}°`);
    // like the firmware: the answer comes when the motor has stopped
    return setTimeout(() => text(res, 200, "Mișcare finalizată!"), Math.min(3000, (steps / speed) * 1000));
  }
  if (url.pathname === "/distance") {
    // a wall about a metre away, a little noise
    return text(res, 200, `${(100 + Math.random() * 5).toFixed(1)} cm`);
  }
  if (url.pathname === "/toggle_en") {
    const m = Number(q.get("motor"));
    state.soft[m] = !state.soft[m];
    return text(res, 200, state.soft[m] ? "MOALE" : "INCORDAT");
  }
  return text(res, 404, "Not found");
}

function drive(req, res) {
  const url = new URL(req.url, "http://x");
  const q = url.searchParams;
  const which = (m) => (m === "all" ? [0, 1, 2, 3] : [Number(m) - 1]).filter((i) => i >= 0 && i < 4);
  const show = () => state.wheels.map((w, i) => `M${i + 1}:${w.running ? (w.dir > 0 ? "+" : "-") + w.speed : "stop"}`).join(" ");
  switch (url.pathname) {
    case "/dir":
      for (const i of which(q.get("m"))) state.wheels[i].dir = Number(q.get("val")) < 0 ? -1 : 1;
      return text(res, 200, "OK");
    case "/speed":
      for (const i of which(q.get("m"))) state.wheels[i].speed = Math.max(0, Math.min(255, Number(q.get("val"))));
      return text(res, 200, "OK");
    case "/start":
    case "/stop":
      for (const i of which(q.get("m"))) state.wheels[i].running = url.pathname === "/start";
      log("drive", show());
      return text(res, 200, "OK");
    case "/status":
      return res.writeHead(200, { "Content-Type": "application/json" }).end(JSON.stringify(state.wheels));
    default:
      return text(res, 404, "Not found");
  }
}

for (const [name, handler, port] of [["cam", camera, PORT], ["motors", motors, PORT + 1], ["drive", drive, PORT + 2]])
  createServer(handler).listen(port, HOST, () => console.log(`${name.padEnd(6)} http://${HOST}:${port}`));
