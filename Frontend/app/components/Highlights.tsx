'use client';
import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Container from '@mui/material/Container';
import Grid from '@mui/material/Grid';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import AutoFixHighRoundedIcon from '@mui/icons-material/AutoFixHighRounded';
import ConstructionRoundedIcon from '@mui/icons-material/ConstructionRounded';
import QueryStatsRoundedIcon from '@mui/icons-material/QueryStatsRounded';
import SettingsSuggestRoundedIcon from '@mui/icons-material/SettingsSuggestRounded';
import SupportAgentRoundedIcon from '@mui/icons-material/SupportAgentRounded';
import ThumbUpAltRoundedIcon from '@mui/icons-material/ThumbUpAltRounded';

const items = [
  {
    icon: <SettingsSuggestRoundedIcon />,
    title: 'Upload Video',
    description:
      'Provide a clear video of your face (MP4, AVI, or WebM format). Works best with good lighting and clear frontal view.',
  },
  {
    icon: <ConstructionRoundedIcon />,
    title: 'Face Detection',
    description:
      'RhythmMamba detects your face region using Haar Cascade classifier. Expands region by 50% to capture full face context.',
  },
  {
    icon: <ThumbUpAltRoundedIcon />,
    title: 'Preprocess',
    description:
      'Crop face region, resize to 128×128 pixels, and normalize color values. Standardize to mean=0, std=1.',
  },
  {
    icon: <AutoFixHighRoundedIcon />,
    title: 'Model Inference',
    description:
      'RhythmMamba processes each chunk through Vision Transformer patching + 12-layer Mamba SSM encoder.',
  },
  {
    icon: <SupportAgentRoundedIcon />,
    title: 'FFT Analysis',
    description:
      'Extract rPPG signal and apply Fast Fourier Transform to find dominant frequency = heart rate.',
  },
  {
    icon: <QueryStatsRoundedIcon />,
    title: 'Display Results',
    description:
      'Show heart rate with confidence score, signal quality, SNR, and rPPG waveform visualization.',
  },
];

export default function Highlights() {
  return (
    <Box
      id="highlights"
      sx={{
        pt: { xs: 4, sm: 12 },
        pb: { xs: 8, sm: 16 },
        color: 'text.primary'
      }}
    >
      <Container
        sx={{
          position: 'relative',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          gap: { xs: 3, sm: 6 },
        }}
      >
        <Box
          sx={{
            width: { sm: '100%', md: '60%' },
            textAlign: { sm: 'left', md: 'center' },
          }}
        >
          <Typography component="h2" variant="h4" gutterBottom>
            How It Works
          </Typography>
          <Typography variant="body1" sx={{ color: 'text.secondary' }}>
            RhythmMamba processes your video in 6 steps: upload → detect → preprocess → infer → analyze → visualize. The entire pipeline is optimized for speed and accuracy.
          </Typography>
        </Box>
        <Grid container spacing={2}>
          {items.map((item, index) => (
            <Grid size={{ xs: 12, sm: 6, md: 4 }} key={index}>
              <Stack
                direction="column"
                component={Card}
                spacing={1}
                useFlexGap
                sx={{
                  color: 'inherit',
                  p: 3,
                  height: '100%',
                  borderColor: (theme) => (theme.vars || theme).palette.divider,
                  backgroundColor: (theme) => (theme.vars || theme).palette.background.paper,
                }}
              >
                <Box sx={{ opacity: '50%' }}>{item.icon}</Box>
                <div>
                  <Typography gutterBottom sx={{ fontWeight: 'medium' }}>
                    {item.title}
                  </Typography>
                  <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                    {item.description}
                  </Typography>
                </div>
              </Stack>
            </Grid>
          ))}
        </Grid>
      </Container>
    </Box>
  );
}
