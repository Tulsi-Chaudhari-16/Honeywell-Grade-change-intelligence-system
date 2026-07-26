import type { Metadata } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import DashboardShell from '@/components/DashboardShell';

const inter = Inter({ subsets: ['latin'] });

export const metadata: Metadata = {
  title: 'GCIS Console - Honeywell',
  description: 'Grade Change Intelligence System',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.className} bg-gray-900 text-gray-100`}>
        <DashboardShell>
          {children}
        </DashboardShell>
      </body>
    </html>
  );
}
