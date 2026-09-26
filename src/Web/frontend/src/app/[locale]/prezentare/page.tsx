import type { ReactNode } from "react";
import type { Metadata } from "next";
import Image from "next/image";
import { getTranslations, setRequestLocale } from "next-intl/server";
import Accordion from "@mui/material/Accordion";
import AccordionDetails from "@mui/material/AccordionDetails";
import AccordionSummary from "@mui/material/AccordionSummary";
import Alert from "@mui/material/Alert";
import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import Chip from "@mui/material/Chip";
import Typography from "@mui/material/Typography";
import ExpandMore from "@mui/icons-material/ExpandMore";
import FlightTakeoffOutlined from "@mui/icons-material/FlightTakeoffOutlined";
import AutoAwesomeOutlined from "@mui/icons-material/AutoAwesomeOutlined";
import MapOutlined from "@mui/icons-material/MapOutlined";
import AssignmentTurnedInOutlined from "@mui/icons-material/AssignmentTurnedInOutlined";
import AccountBalanceOutlined from "@mui/icons-material/AccountBalanceOutlined";
import ApartmentOutlined from "@mui/icons-material/ApartmentOutlined";
import PolicyOutlined from "@mui/icons-material/PolicyOutlined";
import PublicOutlined from "@mui/icons-material/PublicOutlined";
import LockOutlined from "@mui/icons-material/LockOutlined";
import BadgeOutlined from "@mui/icons-material/BadgeOutlined";
import HistoryOutlined from "@mui/icons-material/HistoryOutlined";
import FileDownloadOutlined from "@mui/icons-material/FileDownloadOutlined";
import TranslateOutlined from "@mui/icons-material/TranslateOutlined";
import CheckCircleOutline from "@mui/icons-material/CheckCircleOutlineOutlined";
import ArrowForward from "@mui/icons-material/ArrowForward";
import { DemoButton } from "@/components/landing/DemoButton";
import { LandingHeader } from "@/components/landing/LandingHeader";
import { LeadForm } from "@/components/landing/LeadForm";
import { LogoMark } from "@/components/shell/Logo";
import { LinkButton } from "@/components/common/links";
import { routing } from "@/i18n/routing";
import { getSummary } from "@/lib/data";
import { makeFormat } from "@/lib/format";

type Item = { title: string; text: string };

/** Sources of every figure quoted on the page (src/Web/docs/PREZENTARE_CERCETARE.md). */
const SOURCES = [
  { label: "ONVV — suprafața viilor și defrișările (agrobiznes.md, 24.09.2026)", url: "https://agrobiznes.md/viile-moldovei-se-restrang-pentru-fiecare-hectar-plantat-alte-sapte-sunt-defrisate.html" },
  { label: "Registrul vitivinicol vs BNS, 2024 (agroexpert.md)", url: "https://agroexpert.md/rom/preturi-si-tendinte/suprafata-vitei-de-vie-din-rm-descreste-la-1-hectar-plantat-5-sunt-defrisate" },
  { label: "Comisia Europeană, Raportul 2025 privind Republica Moldova, capitolul 11", url: "https://enlargement.ec.europa.eu/document/download/23fa6af0-89b3-4532-a3d9-d1638727d14c_en?filename=moldova-report-2025.pdf" },
  { label: "Curtea de Conturi, auditul subvențiilor agricole (bani.md, ianuarie 2026)", url: "https://bani.md/cutremur-financiar-in-agricultura-curtea-de-conturi-statul-a-promis-subventii-de-3-miliarde-fara-bani-in-buget/" },
  { label: "MAIA — terenurile agricole abandonate, Codul funciar nr. 22/2024", url: "https://maia.gov.md/ro/content/6008" },
  { label: "Achiziții de valoare mică — întrebări frecvente AAP", url: "https://tender.gov.md/ro/content/%C3%AEntreb%C4%83ri-adresate-frecvent" },
  { label: "Legea nr. 195/2024 privind protecția datelor cu caracter personal (datepersonale.md)", url: "https://datepersonale.md/legea-nr-195-2024-privind-protectia-datelor-cu-caracter-personal-principalele-prevederi-si-noutati-legislative/" },
];

const MAX = 1200;
const Section = ({ id, children, dark = false, muted = false }: { id?: string; children: ReactNode; dark?: boolean; muted?: boolean }) => (
  <Box
    component="section"
    id={id}
    sx={{
      scrollMarginTop: 80,
      py: { xs: 14, md: 20 },
      bgcolor: dark ? "var(--solemtrix-night)" : muted ? "background.default" : "background.paper",
      color: dark ? "common.white" : "text.primary",
    }}
  >
    <Box sx={{ maxWidth: MAX, mx: "auto", px: { xs: 4, md: 6 } }}>{children}</Box>
  </Box>
);

const Heading = ({ title, subtitle, center = false }: { title: string; subtitle?: string; center?: boolean }) => (
  <Box sx={{ maxWidth: 760, mb: { xs: 8, md: 12 }, mx: center ? "auto" : 0, textAlign: center ? "center" : "left" }}>
    <Typography component="h2" sx={{ fontSize: { xs: 28, md: 38 }, fontWeight: 700, letterSpacing: -0.6, lineHeight: 1.15 }}>
      {title}
    </Typography>
    {subtitle && (
      <Typography sx={{ mt: 3, fontSize: { xs: 17, md: 19 }, lineHeight: 1.55, opacity: 0.8 }}>{subtitle}</Typography>
    )}
  </Box>
);

/** A screenshot in a light browser frame. */
const Shot = ({ src, alt, priority = false }: { src: string; alt: string; priority?: boolean }) => (
  <Box sx={{ borderRadius: 3, overflow: "hidden", boxShadow: 8, border: 1, borderColor: "divider", bgcolor: "background.paper" }}>
    <Box sx={{ height: 28, px: 3, display: "flex", alignItems: "center", gap: 1.5, bgcolor: "grey.100", borderBottom: 1, borderColor: "divider" }}>
      {[0, 1, 2].map((i) => (
        <Box key={i} sx={{ width: 10, height: 10, borderRadius: "50%", bgcolor: "grey.300" }} />
      ))}
    </Box>
    <Image src={src} alt={alt} width={1600} height={1000} priority={priority} sizes="(max-width: 900px) 100vw, 640px" style={{ width: "100%", height: "auto", display: "block" }} />
  </Box>
);

/** The mobile screenshot in a phone outline. */
const Phone = ({ src, alt }: { src: string; alt: string }) => (
  <Box sx={{ mx: "auto", width: { xs: 240, md: 280 }, p: 2, borderRadius: 7, bgcolor: "grey.900", boxShadow: 8 }}>
    <Image src={src} alt={alt} width={780} height={1688} sizes="280px" style={{ width: "100%", height: "auto", display: "block", borderRadius: 20 }} />
  </Box>
);

export async function generateMetadata({ params }: PageProps<"/[locale]/prezentare">): Promise<Metadata> {
  const { locale } = await params;
  const t = await getTranslations({ locale, namespace: "landing.meta" });
  return {
    title: t("title"),
    description: t("description"),
    openGraph: { title: t("title"), description: t("description"), images: [{ url: "/marketing/hero-block.webp", width: 1600, height: 1000 }], type: "website" },
    alternates: { languages: Object.fromEntries(routing.locales.map((l) => [l, l === routing.defaultLocale ? "/" : `/${l}`])) },
  };
}

/** The presentation site: what "/" shows to visitors without a session (src/proxy.ts), also at /prezentare. */
export default async function PresentationPage({ params }: PageProps<"/[locale]/prezentare">) {
  const { locale } = await params;
  setRequestLocale(locale);
  const t = await getTranslations("landing");
  const f = makeFormat(locale);
  // pilot figures come from the served survey itself; the 2-tile test data is not shown as a pilot
  const summary = await getSummary().catch(() => null);
  const pilot = summary && !summary.survey.mock ? summary : null;

  const problems = t.raw("problem.items") as (Item & { value: string; source: string })[];
  const steps = t.raw("how.steps") as Item[];
  const products = t.raw("product.items") as (Item & { eyebrow: string; alt: string })[];
  const audiences = t.raw("audiences.items") as { title: string; who: string; bullets: string[] }[];
  const security = t.raw("security.items") as Item[];
  const engagement = t.raw("engagement.steps") as (Item & { tag: string })[];
  const faq = t.raw("faq.items") as { q: string; a: string }[];
  const chips = t.raw("hero.chips") as string[];

  const stepIcons = [<FlightTakeoffOutlined key="0" />, <AutoAwesomeOutlined key="1" />, <MapOutlined key="2" />, <AssignmentTurnedInOutlined key="3" />];
  const audienceIcons = [<AccountBalanceOutlined key="0" />, <ApartmentOutlined key="1" />, <PolicyOutlined key="2" />];
  const securityIcons = [<PublicOutlined key="0" />, <LockOutlined key="1" />, <BadgeOutlined key="2" />, <HistoryOutlined key="3" />, <FileDownloadOutlined key="4" />, <TranslateOutlined key="5" />];
  const productImages = ["/marketing/hero-row.webp", "/marketing/tasks.webp", "/marketing/mobile-map.webp", "/marketing/dashboard.webp"];

  const stats = pilot && [
    { value: f.num(pilot.survey.surveyed_area_ha, 1), label: t("pilot.area") },
    { value: f.int(pilot.totals.plant_count), label: t("pilot.plants") },
    { value: f.int(pilot.totals.row_count), label: t("pilot.rows") },
    { value: f.num(pilot.totals.row_length_m / 1000, 1), label: t("pilot.rowLength") },
    { value: f.int(pilot.totals.disrupted_rows), label: t("pilot.disrupted") },
    { value: f.int(pilot.totals.target_count), label: t("pilot.targets") },
    { value: f.num(pilot.route.length_m / 1000, 1), label: t("pilot.route") },
  ];

  return (
    <Box sx={{ bgcolor: "background.paper" }}>
      <LandingHeader />
      <main>
        {/* hero */}
        <Box
          sx={{
            color: "common.white",
            bgcolor: "var(--solemtrix-night)",
            backgroundImage:
              "radial-gradient(circle at 80% 30%, var(--mui-palette-primary-dark) 0%, transparent 45%), radial-gradient(circle at 10% 100%, var(--mui-palette-primary-main) 0%, transparent 35%)",
            overflow: "hidden",
          }}
        >
          <Box
            sx={{
              maxWidth: MAX,
              mx: "auto",
              px: { xs: 4, md: 6 },
              py: { xs: 14, md: 20 },
              display: "grid",
              gridTemplateColumns: { xs: "1fr", md: "1fr 1.1fr" },
              gap: { xs: 10, md: 12 },
              alignItems: "center",
            }}
          >
            <Box>
              <Typography sx={{ color: "primary.light", fontWeight: 600, letterSpacing: 0.4, fontSize: 15 }}>{t("hero.eyebrow")}</Typography>
              <Typography component="h1" sx={{ mt: 3, fontSize: { xs: 36, md: 52 }, fontWeight: 800, lineHeight: 1.08, letterSpacing: -1.2 }}>
                {t("hero.title")}
              </Typography>
              <Typography sx={{ mt: 5, fontSize: { xs: 17, md: 19 }, lineHeight: 1.6, opacity: 0.82 }}>{t("hero.subtitle")}</Typography>
              <Box sx={{ mt: 8, display: "flex", flexWrap: "wrap", gap: 3 }}>
                <Button href="#contact" variant="contained" size="large" endIcon={<ArrowForward />} sx={{ py: 3, px: 6, fontSize: 16 }}>
                  {t("hero.ctaPilot")}
                </Button>
                <DemoButton variant="outlined" size="large" sx={{ py: 3, px: 6, fontSize: 16, color: "common.white", borderColor: "grey.500" }}>
                  {t("hero.ctaDemo")}
                </DemoButton>
              </Box>
              <Box sx={{ mt: 8, display: "flex", flexWrap: "wrap", gap: 2 }}>
                {chips.map((c) => (
                  <Box
                    key={c}
                    sx={{ display: "inline-flex", alignItems: "center", gap: 1.5, px: 3, py: 1.5, borderRadius: 999, border: 1, borderColor: "grey.600", fontSize: 14 }}
                  >
                    <CheckCircleOutline sx={{ fontSize: 18, color: "primary.light" }} />
                    {c}
                  </Box>
                ))}
              </Box>
            </Box>
            <Shot src="/marketing/hero-block.webp" alt={t("hero.imageAlt")} priority />
          </Box>
        </Box>

        {/* pilot figures */}
        {stats && pilot && (
          <Section muted>
            <Typography component="h2" sx={{ fontSize: { xs: 22, md: 26 }, fontWeight: 700 }}>
              {t("pilot.title")}
            </Typography>
            <Typography color="text.secondary" sx={{ mt: 2, mb: 8 }}>
              {t("pilot.caption", { date: f.date(pilot.survey.captured_at), tiles: f.int(pilot.survey.tiles_total) })}
            </Typography>
            <Box
              sx={{
                display: "grid",
                gridTemplateColumns: { xs: "1fr 1fr", sm: "repeat(4, 1fr)", lg: "repeat(7, 1fr)" },
                gap: 4,
                // 7 figures: the last one takes the free row on phones
                "& > :last-child": { gridColumn: { xs: "1 / -1", sm: "auto" } },
              }}
            >
              {stats.map((s) => (
                <Box key={s.label} sx={{ p: 4, borderRadius: 2, bgcolor: "background.paper", boxShadow: 1 }}>
                  <Typography sx={{ fontSize: { xs: 26, md: 30 }, fontWeight: 800, color: "primary.main", letterSpacing: -0.5 }}>{s.value}</Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mt: 1, lineHeight: 1.35 }}>
                    {s.label}
                  </Typography>
                </Box>
              ))}
            </Box>
          </Section>
        )}

        {/* problem */}
        <Section id="problem">
          <Heading title={t("problem.title")} subtitle={t("problem.subtitle")} />
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" }, gap: 5 }}>
            {problems.map((p) => (
              <Box key={p.title} sx={{ p: { xs: 6, md: 8 }, borderRadius: 3, border: 1, borderColor: "divider" }}>
                <Typography sx={{ fontSize: { xs: 28, md: 34 }, fontWeight: 800, color: "warning.dark", letterSpacing: -0.6 }}>{p.value}</Typography>
                <Typography component="h3" sx={{ mt: 2, fontSize: 19, fontWeight: 700 }}>
                  {p.title}
                </Typography>
                <Typography color="text.secondary" sx={{ mt: 2, lineHeight: 1.6 }}>
                  {p.text}
                </Typography>
                <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 4 }}>
                  {p.source}
                </Typography>
              </Box>
            ))}
          </Box>
        </Section>

        {/* how it works */}
        <Section id="how" muted>
          <Heading title={t("how.title")} subtitle={t("how.subtitle")} />
          <Box component="ol" sx={{ p: 0, m: 0, listStyle: "none", display: "grid", gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", lg: "repeat(4, 1fr)" }, gap: 5 }}>
            {steps.map((s, i) => (
              <Box component="li" key={s.title} sx={{ p: 6, borderRadius: 3, bgcolor: "background.paper", boxShadow: 1 }}>
                <Box sx={{ display: "flex", alignItems: "center", gap: 3, color: "primary.main", "& svg": { fontSize: 30 } }}>
                  {stepIcons[i]}
                  <Typography sx={{ fontWeight: 700, color: "text.secondary" }}>{`0${i + 1}`}</Typography>
                </Box>
                <Typography component="h3" sx={{ mt: 4, fontSize: 19, fontWeight: 700 }}>
                  {s.title}
                </Typography>
                <Typography color="text.secondary" sx={{ mt: 2, lineHeight: 1.6 }}>
                  {s.text}
                </Typography>
              </Box>
            ))}
          </Box>
        </Section>

        {/* product */}
        <Section>
          <Heading title={t("product.title")} />
          <Box sx={{ display: "flex", flexDirection: "column", gap: { xs: 14, md: 20 } }}>
            {products.map((p, i) => (
              <Box
                key={p.title}
                sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: i % 2 ? "1.2fr 1fr" : "1fr 1.2fr" }, gap: { xs: 8, md: 12 }, alignItems: "center" }}
              >
                <Box sx={{ order: { md: i % 2 ? 2 : 1 } }}>
                  <Typography sx={{ color: "primary.main", fontWeight: 700, textTransform: "uppercase", letterSpacing: 1, fontSize: 13 }}>{p.eyebrow}</Typography>
                  <Typography component="h3" sx={{ mt: 2, fontSize: { xs: 24, md: 30 }, fontWeight: 700, letterSpacing: -0.4, lineHeight: 1.2 }}>
                    {p.title}
                  </Typography>
                  <Typography color="text.secondary" sx={{ mt: 4, fontSize: 17, lineHeight: 1.65 }}>
                    {p.text}
                  </Typography>
                </Box>
                <Box sx={{ order: { md: i % 2 ? 1 : 2 } }}>
                  {productImages[i].includes("mobile") ? <Phone src={productImages[i]} alt={p.alt} /> : <Shot src={productImages[i]} alt={p.alt} />}
                </Box>
              </Box>
            ))}
          </Box>
        </Section>

        {/* audiences */}
        <Section id="audiences" muted>
          <Heading title={t("audiences.title")} subtitle={t("audiences.subtitle")} />
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" }, gap: 5 }}>
            {audiences.map((a, i) => (
              <Box key={a.title} sx={{ p: 7, borderRadius: 3, bgcolor: "background.paper", boxShadow: 1 }}>
                <Box sx={{ color: "primary.main", "& svg": { fontSize: 34 } }}>{audienceIcons[i]}</Box>
                <Typography component="h3" sx={{ mt: 4, fontSize: 21, fontWeight: 700 }}>
                  {a.title}
                </Typography>
                <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                  {a.who}
                </Typography>
                <Box component="ul" sx={{ mt: 5, mb: 0, pl: 0, listStyle: "none", display: "flex", flexDirection: "column", gap: 3 }}>
                  {a.bullets.map((b) => (
                    <Box component="li" key={b} sx={{ display: "flex", gap: 2 }}>
                      <CheckCircleOutline fontSize="small" color="success" sx={{ mt: 0.3 }} />
                      <Typography sx={{ lineHeight: 1.5 }}>{b}</Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            ))}
          </Box>
        </Section>

        {/* security */}
        <Section id="security" dark>
          <Heading title={t("security.title")} subtitle={t("security.subtitle")} />
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr", lg: "repeat(3, 1fr)" }, gap: 8 }}>
            {security.map((s, i) => (
              <Box key={s.title}>
                <Box sx={{ color: "primary.light", "& svg": { fontSize: 30 } }}>{securityIcons[i]}</Box>
                <Typography component="h3" sx={{ mt: 3, fontSize: 18, fontWeight: 700 }}>
                  {s.title}
                </Typography>
                <Typography sx={{ mt: 2, lineHeight: 1.6, opacity: 0.78 }}>{s.text}</Typography>
              </Box>
            ))}
          </Box>
          <Typography sx={{ mt: 12, pt: 6, borderTop: 1, borderColor: "grey.700", opacity: 0.78, lineHeight: 1.6 }}>{t("security.note")}</Typography>
        </Section>

        {/* engagement */}
        <Section id="engagement">
          <Heading title={t("engagement.title")} subtitle={t("engagement.subtitle")} />
          <Box component="ol" sx={{ p: 0, m: 0, listStyle: "none", display: "grid", gridTemplateColumns: { xs: "1fr", md: "repeat(3, 1fr)" }, gap: 5 }}>
            {engagement.map((e, i) => (
              <Box
                component="li"
                key={e.title}
                sx={{ p: 7, borderRadius: 3, border: i === 0 ? 2 : 1, borderColor: i === 0 ? "primary.main" : "divider", position: "relative" }}
              >
                <Chip size="small" label={e.tag} color={i === 0 ? "primary" : "default"} />
                <Typography component="h3" sx={{ mt: 4, fontSize: 21, fontWeight: 700 }}>
                  {`${i + 1}. ${e.title}`}
                </Typography>
                <Typography color="text.secondary" sx={{ mt: 2, lineHeight: 1.6 }}>
                  {e.text}
                </Typography>
              </Box>
            ))}
          </Box>
          <Alert severity="info" sx={{ mt: 8 }}>
            {t("engagement.budgetNote")}
          </Alert>
        </Section>

        {/* faq */}
        <Section id="faq" muted>
          <Heading title={t("faq.title")} />
          <Box sx={{ maxWidth: 880 }}>
            {faq.map((q) => (
              <Accordion key={q.q} disableGutters elevation={0} sx={{ border: 1, borderColor: "divider", "&:not(:last-of-type)": { borderBottom: 0 }, "&::before": { display: "none" } }}>
                <AccordionSummary expandIcon={<ExpandMore />} sx={{ px: 5, py: 1 }}>
                  <Typography component="h3" sx={{ fontWeight: 600, fontSize: 17 }}>
                    {q.q}
                  </Typography>
                </AccordionSummary>
                <AccordionDetails sx={{ px: 5, pb: 5 }}>
                  <Typography color="text.secondary" sx={{ lineHeight: 1.65 }}>
                    {q.a}
                  </Typography>
                </AccordionDetails>
              </Accordion>
            ))}
          </Box>
        </Section>

        {/* contact */}
        <Section id="contact">
          <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1.4fr" }, gap: { xs: 8, md: 14 } }}>
            <Heading title={t("contact.title")} subtitle={t("contact.subtitle")} />
            <LeadForm />
          </Box>
        </Section>
      </main>

      {/* footer */}
      <Box component="footer" sx={{ bgcolor: "var(--solemtrix-night)", color: "common.white", py: 14 }}>
        <Box sx={{ maxWidth: MAX, mx: "auto", px: { xs: 4, md: 6 }, display: "grid", gridTemplateColumns: { xs: "1fr", md: "1.3fr 1fr" }, gap: 10 }}>
          <Box>
            <Box sx={{ display: "flex", alignItems: "center", gap: 3 }}>
              <LogoMark size={36} />
              <Typography sx={{ fontSize: 24, fontWeight: 700 }}>solemtrix</Typography>
            </Box>
            <Typography sx={{ mt: 4, opacity: 0.8, maxWidth: 460, lineHeight: 1.6 }}>{t("footer.tagline")}</Typography>
            <Typography variant="body2" sx={{ mt: 4, opacity: 0.65 }}>
              {t("footer.hackathon")}
            </Typography>
            <Typography variant="caption" component="p" sx={{ mt: 2, opacity: 0.55 }}>
              {t("footer.attribution")}
            </Typography>
          </Box>
          <Box>
            <Typography sx={{ fontWeight: 700 }}>{t("footer.app")}</Typography>
            <Box sx={{ mt: 3, display: "flex", flexWrap: "wrap", gap: 3 }}>
              <LinkButton href="/login" variant="outlined" sx={{ color: "common.white", borderColor: "grey.600" }}>
                {t("footer.signIn")}
              </LinkButton>
              <DemoButton variant="text" sx={{ color: "primary.light" }}>
                {t("footer.demo")}
              </DemoButton>
            </Box>
            <Box component="details" sx={{ mt: 8, "& summary": { cursor: "pointer", fontWeight: 600, opacity: 0.85 } }}>
              <summary>{t("footer.sources")}</summary>
              <Box component="ol" sx={{ mt: 3, pl: 5, display: "flex", flexDirection: "column", gap: 2 }}>
                {SOURCES.map((s) => (
                  <Typography component="li" variant="body2" key={s.url} sx={{ opacity: 0.75 }}>
                    <Box component="a" href={s.url} target="_blank" rel="noopener noreferrer" sx={{ color: "inherit" }}>
                      {s.label}
                    </Box>
                  </Typography>
                ))}
              </Box>
            </Box>
          </Box>
        </Box>
      </Box>
    </Box>
  );
}
