import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

/** Solemtrix mark: a sun arc over a grid of vine rows. */
export function LogoMark({ size = 32 }: { size?: number }) {
  return (
    <Box
      component="svg"
      viewBox="0 0 32 32"
      sx={{ width: size, height: size, flexShrink: 0, color: "primary.main" }}
      aria-hidden
    >
      <rect x="1" y="1" width="30" height="30" rx="8" fill="currentColor" />
      <path d="M8 14a8 8 0 0 1 16 0" fill="none" stroke="white" strokeWidth="2.4" strokeLinecap="round" />
      <g stroke="white" strokeWidth="2" strokeLinecap="round" opacity="0.9">
        <path d="M8 19h16" />
        <path d="M8 23h16" strokeDasharray="3 2.2" />
        <path d="M8 27h16" />
      </g>
    </Box>
  );
}

export function Logo({ collapsed = false }: { collapsed?: boolean }) {
  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 2.5 }}>
      <LogoMark />
      {!collapsed && (
        <Typography sx={{ fontSize: 22, fontWeight: 700, letterSpacing: -0.4, color: "text.primary" }}>
          solemtrix
        </Typography>
      )}
    </Box>
  );
}
