'use client';

import React from 'react';
import {
  Box,
  Button,
  Container,
  MenuItem,
  Select,
  Stack,
  Typography,
  Grid,
} from '@mui/material';
import { styled, useTheme } from '@mui/material/styles';
import { useVitals } from '@/context/VitalsContext';


const StyledBox = styled('div')(({ theme }) => ({
  alignSelf: 'center',
  width: '100%',
  maxWidth: "100%",


  height: '100%',

  marginTop: theme.spacing(8),
  borderRadius: (theme.vars || theme).shape.borderRadius,
  padding: theme.spacing(3),

  background: 'linear-gradient(135deg, #f8faff, #eef2ff)',
  backdropFilter: 'blur(12px)',
  
  outline: '6px solid',
  outlineColor: 'hsla(220, 25%, 80%, 0.2)',
  border: '1px solid',
  borderColor: (theme.vars || theme).palette.grey[200],
  boxShadow: '0 0 12px 8px hsla(220, 25%, 80%, 0.2), 0 12px 40px rgba(0,0,0,0.12)',
  backgroundSize: 'cover',

  [theme.breakpoints.up('md')]: {
    marginTop: theme.spacing(10),
    minHeight: 600,
    display: 'flex',
    flexDirection: 'column', 
  },

  ...theme.applyStyles('dark', {
    background: 'linear-gradient(135deg, rgba(20,24,40,0.95), rgba(10,12,24,0.95))',
    boxShadow: '0 0 24px 12px hsla(210, 100%, 25%, 0.2), 0 30px 80px rgba(0,0,0,0.6)',
    outlineColor: 'hsla(220, 20%, 42%, 0.1)',
    borderColor: (theme.vars || theme).palette.grey[700],
  }),
}));


export default function Hero() {
  const theme = useTheme();
  
  const { 
    vitals, 
    runInference, 
    selectFile, 
    setModelChoice, 
    clearVitals 
  } = useVitals();

  const fileInputRef = React.useRef<HTMLInputElement | null>(null);

  const handleVideoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      selectFile(e.target.files[0]);
    }
    e.target.value = '';
  };

  const handleRunAnalysis = async () => {
    await runInference();
  };

  const handleBoxClick = () => {
    if (fileInputRef.current) {
      fileInputRef.current.click();
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      selectFile(e.dataTransfer.files[0]);
    }
  };

  const displayValue = (val: number | null) => {
    return val !== null ? `${Math.round(val * 100) / 100}` : '--';
  };

  return (
    <Box
      id="hero"
      sx={(theme) => ({
        width: '100%',
        backgroundRepeat: 'no-repeat',
        backgroundImage:
          'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 90%), transparent)',
        ...theme.applyStyles('dark', {
          backgroundImage:
            'radial-gradient(ellipse 80% 50% at 50% -20%, hsl(210, 100%, 16%), transparent)',
        }),
      })}
    >
      <Container
        sx={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          pt: { xs: 14, sm: 20 },
          pb: { xs: 8, sm: 12 },
        }}
      >
        <Stack spacing={2} sx={{ alignItems: 'center', width: { xs: '100%', sm: '70%' } }}>
          <Typography
            variant="h1"
            sx={{
              textAlign: 'center',
              fontSize: 'clamp(2.5rem, 8vw, 3.5rem)',
              fontWeight: 700,
              color: 'primary.main',
            }}
          >
            Extract Health Vitals from Videos using AI.
          </Typography>

          <Typography
            sx={{
              textAlign: 'center',
              color: 'text.secondary',
              width: { sm: '100%', md: '80%' },
            }}
          >
            Rhythm Mamba uses remote photoplethysmography (rPPG) to read your heart rate and other vitals from a simple video.
          </Typography>
        </Stack>

        <StyledBox id="image">
          <Box
            sx={{
              display: "flex",
              flexDirection: "column",
              height: "100%",
              gap: 3,
              width: "100%",
            }}
          >
            <Box
              sx={{
                bgcolor: "background.paper",
                borderRadius: 4,
                p: { xs: 2, md: 3 },
                boxShadow: 3,
                textAlign: "center",
                flexShrink: 0,
                width: "100%",
              }}
            >
              <Typography variant="h5" fontWeight={800} color="text.primary" sx={{ mb: 1 }}>
                Upload a Video to Begin
              </Typography>
              <Typography color="text.secondary" mb={3}>
                Upload your video to extract vitals using our trained RhythmMamba model.
              </Typography>

              <Grid container spacing={2} sx={{ width: "100%", margin: 0 }}>
                {[
                  { label: "HEART RATE", value: displayValue(vitals.hr), unit: "bpm", desc: "Extracted via rPPG" },
                  { label: "SIGNAL QUALITY", value: displayValue(vitals.snr), unit: "dB", desc: "Signal-to-noise ratio" },
                  { label: "QUALITY SCORE", value: displayValue(vitals.quality ? vitals.quality * 100 : null), unit: "%", desc: "Overall confidence" },
                ].map((m) => (
                  <Grid 
                    key={m.label} 
                    size={{ xs: 12, md: 4 }}
                    sx={{ display: 'flex', pl: { xs: 0, md: 2 }, pt: { xs: 2, md: 0 } }} 
                  >
                    <Box
                      sx={{
                        bgcolor: "background.default",
                        borderRadius: 3,
                        p: 2,
                        width: "100%",
                        textAlign: "center",
                        display: "flex",
                        flexDirection: "column",
                        justifyContent: "center",
                        border: '1px solid',
                        borderColor: 'divider',
                      }}
                    >
                      <Typography variant="caption" fontWeight={700} color="text.secondary">
                        {m.label}
                      </Typography>
                      <Typography variant="h4" fontWeight={800} color="text.primary" my={0.5}>
                        {m.value} <span style={{ fontSize: 16 }}>{m.unit}</span>
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        {m.desc}
                      </Typography>
                    </Box>
                  </Grid>
                ))}
              </Grid>
            </Box>

            <Box
              sx={{
                display: "flex",
                flexDirection: { xs: "column", md: "row" },
                justifyContent: "space-between",
                alignItems: "center",
                gap: 2,
                flexShrink: 0,
                width: "100%",
              }}
            >
              <Box sx={{ 
                display: "flex", 
                alignItems: "center", 
                gap: 2, 
                width: { xs: "100%", md: "auto" },
                justifyContent: { xs: "space-between", md: "flex-start" } 
              }}>
                <Typography fontWeight={700} color="text.primary">Model</Typography>
                <Select 
                  size="small" 
                  value={vitals.modelChoice} 
                  onChange={(e) => setModelChoice(e.target.value as 'prebuilt' | 'self')} 
                  sx={{ minWidth: 200 }}
                >
                  <MenuItem value="prebuilt">Prebuilt (recommended)</MenuItem>
                  <MenuItem value="self">Self-Trained</MenuItem>
                </Select>
              </Box>

              <Box sx={{ 
                display: "flex", 
                gap: 1.5,
                width: { xs: "100%", md: "auto" },
                flexDirection: { xs: "column", sm: "row" } 
              }}>
                
                <Button 
                  variant="outlined" 
                  disabled={!vitals.uploadedFile || vitals.isLoading} 
                  onClick={handleRunAnalysis} 
                  color="info" 
                  fullWidth 
                  sx={{ whiteSpace: "nowrap" }}
                >
                  {vitals.isLoading ? 'Analyzing...' : 'Run analysis'}
                </Button>

                {vitals.uploadedFile && (
                  <Button variant="contained" onClick={clearVitals} color="error" fullWidth>
                    Delete
                  </Button>
                )}
              </Box>
            </Box>

            <Box sx={{ flex: 1, minHeight: 0, width: "100%" }}>
              <Grid container spacing={3} sx={{ height: "100%", width: "100%", margin: 0 }}>
                

                <Grid size={{ xs: 12, md: 6 }} sx={{ display: 'flex', flexDirection: 'column', pl: 0, pt: { xs: 2, md: 0 } }}>
                  <Box
                    sx={{
                      bgcolor: "background.paper",
                      borderRadius: 3,
                      p: 3,
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      justifyContent: "center",
                      boxShadow: 3,
                      minHeight: { xs: 200, md: 'auto' },
                      width: "100%"
                    }}
                  >
                    <Typography fontWeight={800} mb={3} color="text.primary" sx={{ textAlign: 'center' }}>
                      Tips for uploading video
                    </Typography>

                    <Box sx={{ width: "100%", display: "flex", justifyContent: "center" }}>
                      <Stack spacing={1.5} color="text.secondary" sx={{ textAlign: "left" }}>
                        <Typography variant="body2">✓ Keep face clearly visible</Typography>
                        <Typography variant="body2">✓ Good lighting recommended</Typography>
                        <Typography variant="body2">✓ 30–60 seconds duration</Typography>
                        <Typography variant="body2">✓ Avoid rapid movement</Typography>
                      </Stack>
                    </Box>
                  </Box>
                </Grid>

                <Grid size={{ xs: 12, md: 6 }} sx={{ display: 'flex', flexDirection: 'column', pt: { xs: 2, md: 0 } }}>
                  <Box
                    sx={{
                      bgcolor: "background.paper",
                      borderRadius: 3,
                      border: "2px dashed",
                      borderColor: "divider",
                      flex: 1,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      textAlign: "center",
                      boxShadow: 3,
                      p: 2,
                      minHeight: { xs: 200, md: 'auto' },
                      width: "100%",
                      
                      cursor: "pointer", 
                      transition: "border-color 0.2s, background-color 0.2s",
                      "&:hover": { 
                        borderColor: "primary.main",
                        backgroundColor: (theme) => theme.vars 
                          ? `rgba(${theme.vars.palette.primary.mainChannel} / 0.05)` 
                          : "rgba(25, 118, 210, 0.04)"
                      } 
                    }}
                    onClick={handleBoxClick}
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                  >
                    <input
                      type="file"
                      hidden
                      accept="video/*"
                      ref={fileInputRef}
                      onChange={handleVideoUpload}
                    />

                    {!vitals.uploadedFile ? (
                      <Box>
                        <Typography variant="h6" fontWeight={800} color="text.primary">
                          Drop your video here
                        </Typography>
                        <Typography color="text.secondary">
                          or click to upload
                        </Typography>
                      </Box>
                    ) : (
                      <Box>
                        <Typography fontWeight={600} sx={{ wordBreak: 'break-all' }} color="text.secondary">
                          {vitals.uploadedFile.name}
                        </Typography>
                         <Typography variant="body2" color="primary">
                          Click to change
                        </Typography>
                      </Box>
                    )}
                  </Box>
                </Grid>

              </Grid>

              {vitals.error && (
                <Typography color="error" sx={{ mt: 2, textAlign: 'center', fontWeight: 'bold' }}>
                  {vitals.error}
                </Typography>
              )}
            </Box>

          </Box>
        </StyledBox>
      </Container>
    </Box>
  );
}