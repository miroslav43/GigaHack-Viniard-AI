// Builds gigahack.pptx for the GigaHack platform (it asks for a .pptx): every page of the LaTeX deck becomes a
// full-slide picture inside the organisers' own template package (same master, layout, theme and 16:9 size).
// Usage: node make_pptx.mjs <template.pptx>   (needs pdftoppm, unzip and zip; run after building build/gigahack.pdf)
import { execFileSync } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const PDF = path.join(HERE, 'build', 'gigahack.pdf');
const OUT = path.join(HERE, 'gigahack.pptx');
const TEMPLATE = process.argv[2];
const DPI = '300';

if (!TEMPLATE || !fs.existsSync(TEMPLATE)) throw new Error('usage: node make_pptx.mjs <GigaHack_2026_Pitch_Template.pptx>');
if (!fs.existsSync(PDF)) throw new Error(`missing ${PDF}: build the deck first`);

const work = fs.mkdtempSync(path.join(os.tmpdir(), 'gigahack-pptx-'));
const pkg = path.join(work, 'pkg');
execFileSync('unzip', ['-q', TEMPLATE, '-d', pkg]);
execFileSync('pdftoppm', ['-r', DPI, '-png', PDF, path.join(work, 'page')]);
const pages = fs.readdirSync(work).filter((f) => /^page-\d+\.png$/.test(f)).sort();

// drop the template's slides and speaker notes; keep master, layout, theme and presentation settings
for (const dir of ['ppt/slides', 'ppt/notesSlides']) fs.rmSync(path.join(pkg, dir), { recursive: true, force: true });
fs.mkdirSync(path.join(pkg, 'ppt/slides/_rels'), { recursive: true });

const P = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"';
const slideXml = (n) => `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<p:sld ${P}><p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr><p:pic><p:nvPicPr><p:cNvPr id="2" name="Slide ${n}"/><p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="rId2"/><a:stretch><a:fillRect/></a:stretch></p:blipFill><p:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="12192000" cy="6858000"/></a:xfrm><a:prstGeom prst="rect"><a:avLst/></a:prstGeom></p:spPr></p:pic></p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>`;
const slideRels = (n) => `<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/deck-${n}.png"/></Relationships>`;

pages.forEach((file, i) => {
  const n = i + 1;
  fs.copyFileSync(path.join(work, file), path.join(pkg, 'ppt/media', `deck-${n}.png`));
  fs.writeFileSync(path.join(pkg, 'ppt/slides', `slide${n}.xml`), slideXml(n));
  fs.writeFileSync(path.join(pkg, 'ppt/slides/_rels', `slide${n}.xml.rels`), slideRels(n));
});

// presentation relationships: keep everything except the old slides, then add ours
const relsPath = path.join(pkg, 'ppt/_rels/presentation.xml.rels');
const rels = fs.readFileSync(relsPath, 'utf8').replace(/<Relationship [^>]*relationships\/slide"[^>]*\/>/g, '');
const ourRels = pages.map((_, i) => `<Relationship Id="rIdDeck${i + 1}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide${i + 1}.xml"/>`).join('');
fs.writeFileSync(relsPath, rels.replace('</Relationships>', `${ourRels}</Relationships>`));

const presPath = path.join(pkg, 'ppt/presentation.xml');
const ids = pages.map((_, i) => `<p:sldId id="${256 + i}" r:id="rIdDeck${i + 1}"/>`).join('');
fs.writeFileSync(presPath, fs.readFileSync(presPath, 'utf8').replace(/<p:sldIdLst>[\s\S]*?<\/p:sldIdLst>/, `<p:sldIdLst>${ids}</p:sldIdLst>`));

const ctPath = path.join(pkg, '[Content_Types].xml');
let ct = fs.readFileSync(ctPath, 'utf8').replace(/<Override PartName="\/ppt\/(slides|notesSlides)\/[^>]*\/>/g, '');
if (!/Extension="png"/i.test(ct)) ct = ct.replace('<Types ', '<Types ').replace(/(<Types[^>]*>)/, '$1<Default Extension="png" ContentType="image/png"/>');
const overrides = pages.map((_, i) => `<Override PartName="/ppt/slides/slide${i + 1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>`).join('');
fs.writeFileSync(ctPath, ct.replace('</Types>', `${overrides}</Types>`));

// template slide images are no longer referenced
for (const f of fs.readdirSync(path.join(pkg, 'ppt/media'))) if (!f.startsWith('deck-')) fs.rmSync(path.join(pkg, 'ppt/media', f));

fs.rmSync(OUT, { force: true });
execFileSync('zip', ['-q', '-X', '-r', OUT, '[Content_Types].xml', '_rels', 'docProps', 'ppt'], { cwd: pkg });
fs.rmSync(work, { recursive: true, force: true });
console.log(`wrote ${OUT}: ${pages.length} slides`);
