// get the ninja-keys element
const ninja = document.querySelector('ninja-keys');

// add the home and posts menu items
ninja.data = [{
    id: "nav-about",
    title: "about",
    section: "Navigation",
    handler: () => {
      window.location.href = "/";
    },
  },{id: "nav-projects",
          title: "projects",
          description: "A collection of my projects.",
          section: "Navigation",
          handler: () => {
            window.location.href = "/projects/";
          },
        },{id: "nav-cv",
          title: "cv",
          description: "",
          section: "Navigation",
          handler: () => {
            window.location.href = "/cv/";
          },
        },{id: "news-i-graduated-with-my-bachelor-of-technology-in-chemical-engineering-iit-madras",
          title: 'I graduated with my Bachelor of Technology in Chemical Engineering @ IIT Madras....',
          description: "",
          section: "News",},{id: "news-i-am-joining-marketing-analytics-team-at-anheuser-busch-inbev-as-associate-data-scientist",
          title: 'I am joining Marketing Analytics Team at Anheuser-Busch InBev as Associate Data Scientist....',
          description: "",
          section: "News",},{id: "news-i-graduated-with-my-master-of-science-in-computer-science-stony-brook-university",
          title: 'I graduated with my Master of Science in Computer Science @ Stony Brook...',
          description: "",
          section: "News",},{id: "projects-cross-shard-consensus",
          title: 'Cross-Shard Consensus',
          description: "Multi-cluster distributed transaction processor with Paxos consensus and 2-phase commit for atomic cross-shard transactions in Go",
          section: "Projects",handler: () => {
              window.location.href = "/projects/cross-shard-consensus/";
            },},{id: "projects-linear-pbft",
          title: 'Linear PBFT',
          description: "Linearizable Byzantine fault-tolerant consensus protocol implementation in Go",
          section: "Projects",handler: () => {
              window.location.href = "/projects/linear-pbft/";
            },},{id: "projects-llava-meets-alternating-attention",
          title: 'LLaVA meets Alternating Attention',
          description: "",
          section: "Projects",handler: () => {
              window.location.href = "/projects/llava-alternating-attn/";
            },},{id: "projects-optimal-polygon-triangulation",
          title: 'Optimal Polygon Triangulation',
          description: "Optimal polygon triangulation via interval dynamic programming with memoization",
          section: "Projects",handler: () => {
              window.location.href = "/projects/optimal-triangulation/";
            },},{id: "projects-promotion-optimization-engine",
          title: 'Promotion Optimization Engine',
          description: "Hierarchical regression based model and optimization engine for promotion allocation",
          section: "Projects",handler: () => {
              window.location.href = "/projects/promo-optimization-engine/";
            },},{id: "projects-ultimate-tic-tac-toe",
          title: 'Ultimate Tic Tac Toe',
          description: "RL agent playing Ultimate Tic Tac Toe",
          section: "Projects",handler: () => {
              window.location.href = "/projects/ultimate-tic-tac-toe/";
            },},{id: "projects-whisper-accent",
          title: 'Whisper Accent',
          description: "Conditioning via adaptive layer normalization for accent-aware English speech recognition",
          section: "Projects",handler: () => {
              window.location.href = "/projects/whisper-accent/";
            },},{id: "projects-yin-yang-classification",
          title: 'Yin Yang Classification',
          description: "Visualizing Yin Yang data classification using various machine learning models",
          section: "Projects",handler: () => {
              window.location.href = "/projects/yin-yang-classification/";
            },},{
        id: 'social-email',
        title: 'email',
        section: 'Socials',
        handler: () => {
          window.open("mailto:%76%6D%75%72%75%67%61%6E@%63%73.%73%74%6F%6E%79%62%72%6F%6F%6B.%65%64%75", "_blank");
        },
      },{
        id: 'social-linkedin',
        title: 'LinkedIn',
        section: 'Socials',
        handler: () => {
          window.open("https://www.linkedin.com/in/vijayabharathi-murugan", "_blank");
        },
      },{
        id: 'social-github',
        title: 'GitHub',
        section: 'Socials',
        handler: () => {
          window.open("https://github.com/mavleo96", "_blank");
        },
      },{
      id: 'light-theme',
      title: 'Change theme to light',
      description: 'Change the theme of the site to Light',
      section: 'Theme',
      handler: () => {
        setThemeSetting("light");
      },
    },
    {
      id: 'dark-theme',
      title: 'Change theme to dark',
      description: 'Change the theme of the site to Dark',
      section: 'Theme',
      handler: () => {
        setThemeSetting("dark");
      },
    },
    {
      id: 'system-theme',
      title: 'Use system default theme',
      description: 'Change the theme of the site to System Default',
      section: 'Theme',
      handler: () => {
        setThemeSetting("system");
      },
    },];
