import { VitalsProvider } from '@/context/VitalsContext';

export const metadata = {
  title: 'Rhythm Mamba',
  description: 'Remote Photoplethysmography & Vitals Extraction',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body>
        <VitalsProvider>
          {children}
        </VitalsProvider>
      </body>
    </html>
  )
}
