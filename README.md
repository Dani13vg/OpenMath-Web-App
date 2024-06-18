# OpenMath-Web-App

![OpenMath Logo](nobackground.png)

Welcome to **OpenMath-Web-App** - a web application combined with the power of LLMs designed to provide students and learners with a personalized and engaging mathematical learning experience.

## Table of Contents

- [About OpenMath](#about-openmath)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [Contact](#contact)

## About OpenMath

**OpenMath-Web-App** is an AI-powered learning tool designed to democratize math education and literacy by offering personalized learning experiences. Our web application leverages Large Language Models (LLMs) to adapt mathematical explanations, problems, and concepts to the age, interests, and learning styles of students. This approach increases motivation by relating math topics to students' interests and helps reduce math anxiety by adapting to individual learning paces and styles.

## Features

- **Personalized Learning Paths**: Tailored content based on user interests and learning styles.
- **Interactive Explanations**: Engage with math through dynamic and interactive content.
- **Progress Tracking**: Monitor your learning journey with detailed progress reports.
- **Accessible Anywhere**: Use OpenMath on any device with internet access.

## Installation

To get started with OpenMath-Web-App, follow these steps:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/yourusername/OpenMath-Web-App.git
    cd OpenMath-Web-App
    ```

2. **Install Dependencies**

    In your virtual environment:

    ```
    pip install -r requirements
    ```

3. **Set the environemnt variable 'GROQ_API_KEY':**

    For this you will need to get your own api key from the [groq platform](https://console.groq.com/keys) and export it in the following way:

    ```
    export GROQ_API_KEY='YOUR_API_KEY'
    ```

   
5. **Run the Application**

    ```bash
    python web-app/app.py
    ```

6. **Open Your Browser**

    Navigate to `http://localhost:5000` to see OpenMath-Web-App in action.

## Usage

1. **Register an Account**: Sign up with your personal information to tailor your learning experience.
2. **Set Your Preferences**: Customize your profile with your interests and learning preferences.
3. **Start Learning**: Dive into a wide range of math topics with personalized explanations and exercises.
4. **Track Your Progress**: Use the progress tracking feature to monitor your learning journey.

## Screenshots

![Home Page](screenshots/home_page.png)
*Home Page to register and login*

![Profile Section](screenshots/profile_section.png)
*Profile Section where users set up their likes and preferences*

![Learning Page](screenshots/learning_page.png)
*Collection of topics that the users can choose to start learning*

## Contributing

We welcome contributions from the community! If you would like to contribute to OpenMath-Web-App, please follow these steps:

1. **Fork the Repository**

2. **Create a Branch**

    ```bash
    git checkout -b feature/YourFeature
    ```

3. **Commit Your Changes**

    ```bash
    git commit -m "Add Your Feature"
    ```

4. **Push to the Branch**

    ```bash
    git push origin feature/YourFeature
    ```

5. **Create a Pull Request**


## Contact

For any inquiries, suggestions, or feedback, please reach out to us at:

- [**Neil de la Fuente**](https://www.linkedin.com/in/neil-de-la-fuente/) - neildlf@gmail.com
- [**Paula Feliu**](https://www.linkedin.com/in/paula-feliu-criado/) - p.feliu12@gmail.com
- [**Roger Garcia**](https://www.linkedin.com/in/roger-garcia-ca%C3%B1o-94657a153/) - rogergarciacanyo@gmail.com
- [**Daniel Vidal**](https://www.linkedin.com/in/daniel-alejandro-vidal-guerra-21386b266/) - danielvidal130602@gmail.com
- **Ana Sofia Vega** - ansoveloz@gmail.com

Feel free to explore, learn, and contribute to making math education more personalized and engaging for everyone with **OpenMath-Web-App**!
