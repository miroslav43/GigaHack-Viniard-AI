// Presentation-site demo requests (public.lead): values shared by the form and its server action.
export const INSTITUTION_TYPES = ["primarie", "consiliu_raional", "minister_agentie", "asociatie_producatori", "altul"] as const;
export type InstitutionType = (typeof INSTITUTION_TYPES)[number];
