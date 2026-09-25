"use client";

// Client wrappers so Server Components can render MUI buttons/chips as locale-aware links
// (a component reference like `component={Link}` cannot cross the server→client boundary).
import Button, { type ButtonProps } from "@mui/material/Button";
import Chip, { type ChipProps } from "@mui/material/Chip";
import { Link } from "@/i18n/routing";

export function LinkButton({ href, ...props }: Omit<ButtonProps<typeof Link>, "component" | "href"> & { href: string }) {
  return <Button component={Link} href={href} {...props} />;
}

export function LinkChip({ href, ...props }: Omit<ChipProps<typeof Link>, "component" | "clickable" | "href"> & { href: string }) {
  return <Chip component={Link} href={href} clickable {...props} />;
}
