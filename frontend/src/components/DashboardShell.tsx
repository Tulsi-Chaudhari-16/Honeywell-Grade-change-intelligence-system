// frontend/components/DashboardShell.tsx
import React from 'react';
import Link from 'next/link';

export default function DashboardShell({ children }: { children: React.ReactNode }) {
  return (
    <div className="min-h-screen bg-gray-900 text-gray-100 font-sans flex flex-col">
      {/* Top Navbar */}
      <header className="bg-gray-800 border-b border-gray-700 shadow-sm z-10">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between h-16">
            <div className="flex">
              <div className="flex-shrink-0 flex items-center gap-3">
                <div className="w-8 h-8 bg-blue-600 rounded-md flex items-center justify-center font-bold text-white shadow-lg">
                  HW
                </div>
                <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-cyan-300">
                  GCIS Console
                </span>
              </div>
              <nav className="hidden sm:ml-8 sm:flex sm:space-x-4">
                <Link href="/" className="inline-flex items-center px-1 pt-1 border-b-2 border-blue-500 text-sm font-medium text-white">
                  Overview
                </Link>
                <Link href="/live" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300 text-sm font-medium text-gray-300 hover:text-white transition-colors">
                  Live Process
                </Link>
                <Link href="/historical" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300 text-sm font-medium text-gray-300 hover:text-white transition-colors">
                  History Explorer
                </Link>
                <Link href="/recipes" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300 text-sm font-medium text-gray-300 hover:text-white transition-colors">
                  Recipes
                </Link>
                <Link href="/reports" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300 text-sm font-medium text-gray-300 hover:text-white transition-colors">
                  Reports
                </Link>
                <Link href="/chat" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300 text-sm font-medium text-gray-300 hover:text-white transition-colors">
                  Chat
                </Link>
                <Link href="/alerts" className="inline-flex items-center px-1 pt-1 border-b-2 border-transparent hover:border-gray-300 text-sm font-medium text-gray-300 hover:text-white transition-colors">
                  Alerts
                </Link>
              </nav>
            </div>
            <div className="flex items-center gap-4">
              <Link href="/health" className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-900/50 text-green-400 border border-green-800 hover:bg-green-900/70 transition-colors">
                System Healthy
              </Link>
              <div className="h-8 w-8 rounded-full bg-gray-700 border border-gray-600 flex items-center justify-center text-sm font-medium">
                OP
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {children}
      </main>
    </div>
  );
}
