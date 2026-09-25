import type { ReactNode } from "react";
import Box from "@mui/material/Box";
import Typography from "@mui/material/Typography";

export function PageHeader({ title, subtitle, action }: { title: string; subtitle?: ReactNode; action?: ReactNode }) {
  return (
    <Box sx={{ display: "flex", alignItems: { xs: "flex-start", sm: "center" }, justifyContent: "space-between", gap: 4, flexWrap: "wrap", mb: 6 }}>
      <Box>
        <Typography variant="h1" component="h1">
          {title}
        </Typography>
        {subtitle && (
          <Typography variant="body1" color="text.secondary" sx={{ mt: 1 }}>
            {subtitle}
          </Typography>
        )}
      </Box>
      {action}
    </Box>
  );
}

export function Page({ children }: { children: ReactNode }) {
  return <Box sx={{ px: { xs: 4, md: 8 }, py: { xs: 5, md: 7 }, maxWidth: 1440, mx: "auto" }}>{children}</Box>;
}
