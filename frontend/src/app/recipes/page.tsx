"use client";

import React, { useEffect, useState } from 'react';

export default function RecipeExplorerPage() {
  const [recipes, setRecipes] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch('/api/v1/recipes')
      .then(res => res.json())
      .then(data => {
        setRecipes(data);
        setLoading(false);
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, []);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex justify-between items-end border-b border-gray-700 pb-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Recipe Explorer</h1>
          <p className="text-gray-400 mt-1">View paper grade recipes and their target constraints.</p>
        </div>
      </div>

      <div className="bg-gray-800 rounded-xl border border-gray-700 shadow-sm overflow-hidden">
        {loading ? (
          <div className="p-8 text-center text-gray-400">Loading recipes...</div>
        ) : (
          <table className="min-w-full divide-y divide-gray-700">
            <thead className="bg-gray-800/50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Grade Code</th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase tracking-wider">Name</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">Target BW (g/m²)</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">Target Moisture (%)</th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-400 uppercase tracking-wider">Actions</th>
              </tr>
            </thead>
            <tbody className="bg-gray-800 divide-y divide-gray-700">
              {recipes.map((recipe, idx) => (
                <tr key={idx} className="hover:bg-gray-750 transition-colors">
                  <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-blue-400 font-mono">
                    {recipe.code}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-200">
                    {recipe.name}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-mono text-gray-300">
                    {recipe.targets.basis_weight?.toFixed(1) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-sm text-right font-mono text-gray-300">
                    {recipe.targets.moisture?.toFixed(1) || 'N/A'}
                  </td>
                  <td className="px-6 py-4 whitespace-nowrap text-right text-sm font-medium">
                    <button className="text-blue-400 hover:text-blue-300">
                      View Limits →
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
