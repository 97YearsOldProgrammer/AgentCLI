import subprocess
import re
from typing import List, Dict, Any


class Bash:
    """
    An implementation of a tool that executes Bash commands.
    Enforces an allowlist of commands and tracks the working directory.
    """

    def __init__(self, cwd: str, allowed_commands: List[str]):
        """
        Initialize the Bash tool.
        
        Args:
            cwd: The initial working directory
            allowed_commands: List of allowed bash commands
        """
        self.cwd = cwd  # The current working directory
        self._allowed_commands = allowed_commands  # Allowed commands

    def exec_bash_command(self, cmd: str) -> Dict[str, str]:
        """
        Execute the bash command after validating against the allowlist.
        
        Args:
            cmd: The bash command to execute
            
        Returns:
            Dictionary with stdout, stderr, cwd, or error message
        """
        if cmd:
            # Check the allowlist
            allowed = True

            for cmd_part in self._extract_commands(cmd):
                if cmd_part not in self._allowed_commands:
                    allowed = False
                    break

            if not allowed:
                return {"error": "Parts of this command were not in the allowlist."}

            return self._run_bash_command(cmd)
        return {"error": "No command was provided"}

    def to_json_schema(self) -> Dict[str, Any]:
        """
        Convert the function signature to a JSON schema for LLM tool calling.
        
        Returns:
            JSON schema dictionary describing the tool
        """
        return {
            "type": "function",
            "function": {
                "name": "exec_bash_command",
                "description": "Execute a bash command and return stdout/stderr and the working directory",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "cmd": {
                            "type": "string",
                            "description": "The bash command to execute"
                        }
                    },
                    "required": ["cmd"],
                },
            },
        }

    def _extract_commands(self, cmd: str) -> List[str]:
        """
        Extract command names from a bash command string.
        Handles pipes, semicolons, and other command separators.
        
        Args:
            cmd: The full command string
            
        Returns:
            List of individual command names
        """
        # Split on common command separators: |, ;, &&, ||
        # Also handle command substitution $() and backticks
        separators = r'[|;&]|\$\(|`|\|\||&&'
        parts = re.split(separators, cmd)
        
        commands = []
        for part in parts:
            part = part.strip()
            if part:
                # Get the first word (the command name)
                # Handle redirections like >> and >
                part = re.sub(r'>+\s*\S+', '', part).strip()
                if part:
                    cmd_name = part.split()[0] if part.split() else ''
                    if cmd_name:
                        commands.append(cmd_name)
        
        return commands

    def _run_bash_command(self, cmd: str) -> Dict[str, str]:
        """
        Runs the bash command and catches exceptions (if any).
        
        Args:
            cmd: The bash command to execute
            
        Returns:
            Dictionary with stdout, stderr, and current working directory
        """
        stdout = ""
        stderr = ""
        new_cwd = self.cwd

        try:
            # Wrap the command so we can keep track of the working directory.
            wrapped = f"{cmd};echo __END__;pwd"
            result = subprocess.run(
                wrapped, 
                shell=True, 
                cwd=self.cwd,
                capture_output=True, 
                text=True,
                executable="/bin/bash"
            )
            stderr = result.stderr
            # Find the separator marker
            split = result.stdout.split("__END__")
            stdout = split[0].strip()

            # If no output/error at all, inform that the call was successful.
            if not stdout and not stderr:
                stdout = "Command executed successfully, without any output."

            # Get the new working directory, and change it
            new_cwd = split[-1].strip()
            self.cwd = new_cwd
        except Exception as e:
            stdout = ""
            stderr = str(e)

        return {"stdout": stdout, "stderr": stderr, "cwd": new_cwd}


# Default list of allowed commands - safe commands that won't cause damage
LIST_OF_ALLOWED_COMMANDS = [
    # Navigation and listing
    "ls", "cd", "pwd", "find", "tree",
    # File viewing
    "cat", "head", "tail", "less", "more", "wc",
    # Text processing
    "grep", "awk", "sed", "sort", "uniq", "cut", "tr",
    # File operations (safe ones)
    "touch", "mkdir", "cp", "echo",
    # System info
    "df", "du", "free", "uname", "whoami", "date", "hostname",
    # Process info
    "ps", "top", "htop",
    # Network info (read-only)
    "ping", "curl", "wget", "ifconfig", "ip",
    # Compression (read operations)
    "tar", "zip", "unzip", "gzip", "gunzip",
    # Other utilities
    "which", "whereis", "file", "stat", "basename", "dirname",
]