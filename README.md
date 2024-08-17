"""
# TMDP: Teleport Markov Decision Process
![alt text](teleport_demo.gif)
## Project Overview

This repository contains the implementation of the Teleport Markov Decision Process (TMDP), a framework introduced as part of my thesis work titled **"Curriculum Reinforcement Learning through Teleportation: The Teleport MDP"**. This work is a significant contribution to the field of Deep Reinforcement Learning (DRL) and explores the concept of Curriculum Learning (CL) by introducing a novel approach for agent training.

The TMDP framework enhances the exploration capabilities of RL agents by incorporating a teleportation mechanism that allows an agent to be relocated to any state during an episode. This process helps the agent to overcome challenges associated with sparse rewards, high-dimensional spaces, and long-term credit assignment.

The project includes both theoretical contributions and practical implementations, showcasing the effectiveness of TMDP-based curricula through empirical evaluation on well-known RL environments.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

To set up the TMDP framework on your local machine, follow these steps:

1. **Clone the repository:**

   \`\`\`bash
   git clone https://github.com/cris96spa/TMDP.git
   \`\`\`

2. **Navigate to the project directory:**

   \`\`\`bash
   cd TMDP
   \`\`\`

3. **Install the required dependencies:**

   Ensure that you have Python 3.x installed. Then, install the necessary Python packages using pip:

   \`\`\`bash
   pip install -r requirements.txt
   \`\`\`

## Usage

After installation, you can experiment with the TMDP framework by running the provided scripts. The repository includes implementations of key algorithms such as:

- **Teleport Model Policy Iteration (TMPI)**
- **Static Teleport (S-T)**
- **Dynamic Teleport (D-T)**

These algorithms are designed to integrate teleportation-based curricula into standard RL training processes.

## Project Structure

The repository is organized as follows:

- **src/**: Contains the core implementation of the TMDP framework and associated RL algorithms.
- **configs/**: Includes configuration files for running experiments in various environments.
- **data/**: Stores the results and logs of the experiments.
- **docs/**: Contains project documentation and the executive summary of the thesis.
- **tests/**: Includes unit tests for the implemented algorithms.
- **run_experiment.py**: Script to launch experiments using the TMDP framework.

## Contributing

Contributions to the TMDP project are welcome! If you would like to contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch (\`git checkout -b feature-branch\`).
3. Make your changes and commit them (\`git commit -m 'Add some feature'\`).
4. Push to the branch (\`git push origin feature-branch\`).
5. Open a pull request.

Please ensure your code follows the style guidelines provided in the \`CONTRIBUTING.md\` file.

## License

This project is licensed under the MIT License. See the \`LICENSE\` file for more details.

## Acknowledgments

This work was carried out as part of my thesis for the Laurea Magistrale in Computer Science and Engineering at Politecnico di Milano, under the supervision of Prof. Marcello Restelli and co-advisors Dott. Alberto Maria Metelli and Dott. Luca Sabbioni. I would like to thank my advisors for their invaluable guidance and support throughout the project.