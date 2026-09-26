// Architecture diagrams of the AI pipeline (archify HTML, static in public/diagrams; specs in docs/architecture).
export const DIAGRAMS = [
  { slug: "sistem", file: "/diagrams/siret3-01-sistem.html" },
  { slug: "perceptie", file: "/diagrams/siret3-02-perceptie.html" },
  { slug: "randuri", file: "/diagrams/siret3-03-randuri.html" },
  { slug: "deseuri", file: "/diagrams/siret3-04-deseuri.html" },
  { slug: "traseu", file: "/diagrams/siret3-05-traseu.html" },
] as const;

export type DiagramSlug = (typeof DIAGRAMS)[number]["slug"];

export function pickDiagram(slug: string | string[] | undefined) {
  return DIAGRAMS.find((d) => d.slug === slug) ?? DIAGRAMS[0];
}
