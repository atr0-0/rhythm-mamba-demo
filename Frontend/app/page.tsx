'use client';
import CssBaseline from '@mui/material/CssBaseline';
import Divider from '@mui/material/Divider';
import Box from '@mui/material/Box';
import AppTheme from './shared-theme/AppTheme';
import AppAppBar from './components/AppAppBar';
import Hero from './components/Hero';
import Highlights from './components/Highlights';
import Features from './components/Features';
import Footer from './components/Footer';

export default function MarketingPage(props: { disableCustomTheme?: boolean }) {
  return (
    <AppTheme {...props}>
      <CssBaseline enableColorScheme />
      <Box sx={{ bgcolor: (theme) => (theme.vars || theme).palette.background.default, minHeight: '100vh' }}>
        <AppAppBar />
        <Hero />
        <Divider />
        <Highlights />
        <Divider />
        <Features />
        <Divider />
        <Footer />
      </Box>
    </AppTheme>
  );
}
