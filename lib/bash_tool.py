import subprocess as sp
import re


#################### Command Lists ####################

LIST_OF_ALLOWED_COMMANDS = [
    "ls", "cd", "pwd", "find", "tree",
    "cat", "head", "tail", "less", "more", "wc",
    "grep", "awk", "sed", "sort", "uniq", "cut", "tr",
    "touch", "mkdir", "cp", "echo",
    "df", "du", "free", "uname", "whoami", "date", "hostname",
    "ps", "top", "htop",
    "ping", "curl", "wget", "ifconfig", "ip",
    "tar", "zip", "unzip", "gzip", "gunzip",
    "which", "whereis", "file", "stat", "basename", "dirname",
]

LIST_OF_AUTO_EXECUTE_COMMANDS = [
    "ls", "pwd", "whoami", "date", "hostname", "uname",
    "df", "free", "ps",
    "which", "whereis", "file", "stat", "basename", "dirname",
    "echo", "cat", "head", "tail", "wc",
    "tree", "find",
    "ls", "cd", 
    "wget",
]


#################### Bash Tool Class ####################

class Bash:
    """

    Executes Bash commands with allowlist enforcement and directory tracking
    """

    def __init__(self, cwd, allowed_commands, auto_execute_commands=None):
        self.cwd                   = cwd
        self._allowed_commands     = allowed_commands
        self._auto_execute_commands = auto_execute_commands or []

    def exec_bash_command(self, cmd):
        """

        Execute the bash command after validating against the allowlist
        """
        if not cmd:
            return {"error": "No command was provided"}

        for cmd_part in self._extract_commands(cmd):
            if cmd_part not in self._allowed_commands:
                return {"error": "Parts of this command were not in the allowlist"}

        return self._run_bash_command(cmd)

    def is_auto_executable(self, cmd):
        """

        Check if all commands in the string are auto-executable
        """
        commands = self._extract_commands(cmd)
        return all(c in self._auto_execute_commands for c in commands)

    def to_json_schema(self):
        """

        Convert the function signature to a JSON schema for LLM tool calling
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

    def _extract_commands(self, cmd):
        """

        Extract command names from a bash command string
        """
        separators = r'[|;&]|\$\(|`|\|\||&&'
        parts      = re.split(separators, cmd)
        commands   = []

        for part in parts:
            part = part.strip()
            if part:
                part = re.sub(r'>+\s*\S+', '', part).strip()
                if part:
                    cmd_name = part.split()[0] if part.split() else ''
                    if cmd_name:
                        commands.append(cmd_name)

        return commands

    def _run_bash_command(self, cmd):
        """

        Runs the bash command and catches exceptions
        """
        stdout  = ""
        stderr  = ""
        new_cwd = self.cwd

        try:
            wrapped = f"{cmd};echo __END__;pwd"
            result  = sp.run(
                wrapped,
                shell=True,
                cwd=self.cwd,
                capture_output=True,
                text=True,
                executable="/bin/bash"
            )

            stderr = result.stderr
            split  = result.stdout.split("__END__")
            stdout = split[0].strip()

            if not stdout and not stderr:
                stdout = "Command executed successfully, without any output"

            new_cwd  = split[-1].strip()
            self.cwd = new_cwd

        except Exception as e:
            stdout = ""
            stderr = str(e)

        return {"stdout": stdout, "stderr": stderr, "cwd": new_cwd}