'use client';
import Box from '@mui/material/Box';
import Button from '@mui/material/Button';
import Container from '@mui/material/Container';
import IconButton from '@mui/material/IconButton';
import InputLabel from '@mui/material/InputLabel';
import Link from '@mui/material/Link';
import Stack from '@mui/material/Stack';
import TextField from '@mui/material/TextField';
import Typography from '@mui/material/Typography';
import GitHubIcon from '@mui/icons-material/GitHub';
import LinkedInIcon from '@mui/icons-material/LinkedIn';
import TwitterIcon from '@mui/icons-material/X';
import SitemarkIcon from './SitemarkIcon';

function Copyright() {
  return (
    <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
      {'Copyright © '}
      <Link color="text.secondary" href="https://mui.com/">
        Sitemark
      </Link>
      &nbsp;
      {new Date().getFullYear()}
    </Typography>
  );
}

export default function Footer() {
  return (
    <Container
      sx={{
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        gap: { xs: 4, sm: 8 },
        py: { xs: 2, sm: 4 },
        textAlign: { sm: 'center', md: 'left' },
      }}
    >
      <Box
        sx={{
          display: 'flex',
          flexDirection: { xs: 'column', sm: 'row' },
          alignItems: 'center',
          justifyContent: 'center',
          width: '100%',
          gap: 2,
        }}
      >
        <Box sx={{ textAlign: 'left', alignItems: 'flex-start', display: 'flex', flexDirection: 'column', flex: 1 }}>
          <Box>
            <Link color="text.secondary" variant="body2" href="https://www.tu-ilmenau.de/universitaet/fakultaeten/fakultaet-informatik-und-automatisierung/profil/institute-und-fachgebiete/institut-fuer-technische-informatik-und-ingenieurinformatik/fachgebiet-neuroinformatik-und-kognitive-robotik/data-sets-code/pulse-rate-detection-dataset-pure" target="_blank" rel="noopener">
              PURE Dataset
            </Link>
            <Typography sx={{ display: 'inline', mx: 0.5, opacity: 0.5 }}>
              &nbsp;•&nbsp;
            </Typography>
            <Link color="text.secondary" variant="body2" href="https://sites.google.com/view/ybenezeth/ubfcrppg" target="_blank" rel="noopener">
              UBFC-rPPG Dataset
            </Link>
          </Box>
          <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
            Copyright © <Link color="text.secondary" href="https://github.com/zizheng-guo/RhythmMamba" target="_blank" rel="noopener">Rhythm Mamba</Link> {new Date().getFullYear()}
          </Typography>
        </Box>
        <Stack
          direction="row"
          spacing={1}
          useFlexGap
          sx={{ color: 'text.secondary' }}
        >
          <IconButton
            color="inherit"
            size="small"
            href="https://github.com/atr0-0/rhythm-mamba-demo"
            aria-label="GitHub"
            sx={{ alignSelf: 'center' }}
            target="_blank"
            rel="noopener"
          >
            <GitHubIcon />
          </IconButton>
          <IconButton
            color="inherit"
            size="small"
            href="https://www.linkedin.com/in/shubham--tandon/"
            aria-label="LinkedIn"
            sx={{ alignSelf: 'center' }}
            target="_blank"
            rel="noopener"
          >
            <LinkedInIcon />
          </IconButton>
        </Stack>
      </Box>
    </Container>
  );
}
