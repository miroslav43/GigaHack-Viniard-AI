"use client";

import IconButton from "@mui/material/IconButton";
import InputAdornment from "@mui/material/InputAdornment";
import TextField from "@mui/material/TextField";
import Tooltip from "@mui/material/Tooltip";
import AutorenewOutlined from "@mui/icons-material/AutorenewOutlined";

/** 16 random characters from an unambiguous alphabet (crypto RNG). */
export function generatePassword() {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnpqrstuvwxyz23456789-_";
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (b) => alphabet[b % alphabet.length]).join("");
}

export function PasswordField({ value, onChange, label, generateLabel }: { value: string; onChange: (v: string) => void; label: string; generateLabel: string }) {
  return (
    <TextField
      label={label}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      fullWidth
      slotProps={{
        input: {
          sx: { fontFamily: "monospace" },
          endAdornment: (
            <InputAdornment position="end">
              <Tooltip title={generateLabel}>
                <IconButton edge="end" onClick={() => onChange(generatePassword())} aria-label={generateLabel}>
                  <AutorenewOutlined />
                </IconButton>
              </Tooltip>
            </InputAdornment>
          ),
        },
      }}
    />
  );
}

