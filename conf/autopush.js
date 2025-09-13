const simpleGit = require('simple-git');
const git = simpleGit();

async function autoCommitPush() {
  try {
    // Get the status of the repo
    const status = await git.status();

    // Combine modified and not-added files
    const changedFiles = status.modified.concat(status.not_added);

    if (changedFiles.length === 0) {
      console.log('No changes detected.');
      return;
    }

    // Display table of changed files
    const tableData = changedFiles.map((file, index) => ({
      '#': index + 1,
      File: file,
      Status: status.not_added.includes(file) ? 'Added' : 'Modified',
    }));
    console.table(tableData);

    // Stage all changes
    await git.add('.');

    // Create commit message with datetime and changed files
    const now = new Date();
    const datetime = now.toISOString().replace('T', ' ').split('.')[0]; // YYYY-MM-DD HH:MM:SS
    const commitMessage = `Auto-commit ${datetime} - ${changedFiles.join(', ')}`;

    // Commit
    await git.commit(commitMessage);
    console.log('Committed:', commitMessage);

    // Push
    await git.push();
    console.log('Pushed to remote repository.');
  } catch (err) {
    console.error('Error:', err);
  }
}

// Run the function
autoCommitPush();
