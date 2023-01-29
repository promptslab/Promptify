# Contributing

-   [Ways to contribute](#ways-to-contribute)
-   [Contribution Content](#contribution-content)
-   [Contribute using GitHub](#contribute-using-github)
-   [Contribute using Git](#contribute-using-git)

## Ways to contribute

Here are some ways you can contribute to this documentation:

-   To make small changes to an article, [Contribute using GitHub](#contribute-using-github).
-   To make large changes, or changes that involve code, [Contribute using Git](#contribute-using-git).
-   Report documentation bugs via GitHub Issues


## Contribution Content

- Code
- New NLP Task
- Prompts for new task, new domain
- Improve Documentation (Readme, docs, colab files etc)

## Contribute using GitHub

Use GitHub to contribute to this documentation without having to clone the repository to your desktop. This is the easiest way to create a pull request in this repository. Use this method to make a minor change that doesn't involve code changes.

### To Contribute using GitHub

1. Find the file you want to contribute to on GitHub.
2. Once you are on the file in GitHub, sign in to GitHub (get a free account [Join GitHub](https://github.com/join).
3. Choose the **pencil icon** (edit the file in your fork of this project) and make your changes in the **<>Edit file** window.
4. Scroll to the bottom and enter a description.
5. Choose **Propose file change**>**Create pull request**.

You now have successfully submitted a pull request.


## Contribute using Git

To get started contributing to the Promptify repository, which includes prompts, code, and documentation, please refer to this step-by-step guide.

1. Fork the repository to your GitHub account to create a copy of the repository associated with your account, enabling you to make changes without altering the original.

2. To create a new branch in your repository, run the following command (replacing "YOUR_GITHUB_USERNAME" with your actual GitHub username):
```bash
git checkout -b YOUR_GITHUB_USERNAME/feature-name
```
This branch will be where you can make changes and additions.

3. It's now time to implement the modifications you want to make to the repository. You can add new files, modify existing files or delete files that are no longer necessary. Once you have completed these changes, you can stage them by utilizing the command:
```bash
git add .
``` 
This command will stage all the changes that have been made.

4. Commit your changes with a meaningful commit message. This message should briefly describe the changes you've made. You can do this by running the command:
```bash
git commit -m "Commit message"
```

5. Push your changes to the branch you created in step 2. You can do this by running the command:
```bash
git push origin YOUR_GITHUB_USERNAME/feature-name
```

6. Once you have made the necessary changes, navigate to the Promptify repository on GitHub and create a pull request. Provide a clear description of the modifications you have made and request a review and approval of the changes to be merged into the main repository.

7. Before starting with the above steps, it is important to note that you should have a local copy of the repository on your machine. You can clone the repository by running the command:
```bash
git clone https://github.com/promptslab/Promptify.git
```

Thank you for your enthusiasm in contributing to Promptify! We are excited to review your proposed changes and incorporate them into our codebase :heart:
