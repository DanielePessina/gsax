import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'gsax',
  description: 'Global Sensitivity Analysis in JAX',
  base: '/gsax/',

  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'Examples', link: '/examples/basic' },
      { text: 'API', link: '/api/problem' },
    ],

    sidebar: {
      '/guide/': {
        base: '/guide/',
        items: [
          { text: 'Getting Started', link: '/getting-started' },
          { text: 'Methods', link: '/methods' },
          { text: 'Benchmarks', link: '/benchmarks' },
        ],
      },
      '/examples/': {
        base: '/examples/',
        items: [
          { text: 'Basic (Ishigami)', link: '/basic' },
          { text: 'Multi-Output', link: '/multi-output' },
          { text: 'Bootstrap CIs', link: '/bootstrap' },
          { text: 'RS-HDMR', link: '/hdmr' },
        ],
      },
      '/api/': {
        base: '/api/',
        items: [
          { text: 'Problem', link: '/problem' },
          { text: 'Sampling', link: '/sampling' },
          { text: 'Analyze (Sobol)', link: '/analyze' },
          { text: 'HDMR', link: '/hdmr' },
        ],
      },
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/danielepessina/gsax' },
    ],

    search: {
      provider: 'local',
    },

    footer: {
      message: 'Released under the MIT License.',
    },
  },
})
