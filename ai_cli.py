import os
import sys
import json
import asyncio
import yaml
from openai import OpenAI
import uuid
from typing import Dict, Any, List
import subprocess
from coding_app import CodeAnalyzer, CodeReviewer, CodeOptimizer, DocGenerator, CodeGenerator
import logging
from pathlib import Path
import shlex
import re

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AICLI")

# Initialize OpenAI client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

class ProjectManager:
    """Manages project creation and file operations."""
    
    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self.base_dir.mkdir(exist_ok=True)
        self.templates_dir = base_dir / "templates"
        self.templates_dir.mkdir(exist_ok=True)
        self._load_templates()
        
    def _load_templates(self):
        """Load project templates from YAML files."""
        self.templates = {}
        for template_file in self.templates_dir.glob("*.yaml"):
            with open(template_file, "r") as f:
                self.templates[template_file.stem] = yaml.safe_load(f)
    
    def create_project(self, project_type: str, name: str) -> Path:
        """Create a new project with the given type and name."""
        project_id = str(uuid.uuid4())
        project_dir = self.base_dir / project_id
        project_dir.mkdir(exist_ok=True)
        
        # Create project metadata
        metadata = {
            "id": project_id,
            "name": name,
            "type": project_type,
            "created_at": str(asyncio.get_event_loop().time()),
            "files": []
        }
        
        # Save metadata
        with open(project_dir / "metadata.yaml", "w") as f:
            yaml.dump(metadata, f)
            
        return project_dir
    
    def add_file(self, project_dir: Path, filename: str, content: str):
        """Add a file to the project."""
        file_path = project_dir / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, "w") as f:
            f.write(content)
            
        # Update metadata
        metadata_path = project_dir / "metadata.yaml"
        with open(metadata_path, "r") as f:
            metadata = yaml.safe_load(f)
            
        metadata["files"].append({
            "name": filename,
            "path": str(file_path.relative_to(project_dir)),
            "modified_at": str(asyncio.get_event_loop().time())
        })
        
        with open(metadata_path, "w") as f:
            yaml.dump(metadata, f)

class CodebaseManager:
    """Manages knowledge about the codebase, tracking files and their changes."""
    
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        self.files = {}  # Dictionary to store file info
        self.project_type = None
        self.dependencies = set()
        self.modified_files = set()
        self.created_files = set()
        self._scan_project()
        
    def _scan_project(self):
        """Scan the project directory to build initial knowledge."""
        if not self.project_dir.exists():
            return
            
        # Scan all files in the project
        for file_path in self.project_dir.glob('**/*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(self.project_dir)
                self.files[str(rel_path)] = {
                    'path': str(rel_path),
                    'last_modified': file_path.stat().st_mtime,
                    'size': file_path.stat().st_size,
                    'content_hash': self._calculate_hash(file_path),
                    'type': self._determine_file_type(file_path),
                }
                
        # Detect project type
        self._detect_project_type()
        
    def _calculate_hash(self, file_path: Path) -> str:
        """Calculate a hash of the file content."""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return str(hash(content))
        except Exception:
            return ""
            
    def _determine_file_type(self, file_path: Path) -> str:
        """Determine the type of file based on extension."""
        ext = file_path.suffix.lower()
        if ext in ['.py']:
            return 'python'
        elif ext in ['.js', '.jsx']:
            return 'javascript'
        elif ext in ['.ts', '.tsx']:
            return 'typescript'
        elif ext in ['.html', '.htm']:
            return 'html'
        elif ext in ['.css', '.scss', '.sass']:
            return 'stylesheet'
        elif ext in ['.json']:
            return 'json'
        elif ext in ['.md', '.markdown']:
            return 'markdown'
        elif ext in ['.sh', '.bash']:
            return 'shell'
        else:
            return 'unknown'
            
    def _detect_project_type(self):
        """Detect the type of project based on files."""
        files = set(self.files.keys())
        
        # Check for package.json (Node.js)
        if 'package.json' in files:
            try:
                with open(self.project_dir / 'package.json', 'r') as f:
                    import json
                    data = json.load(f)
                    dependencies = data.get('dependencies', {})
                    
                    # Check for specific frameworks
                    if 'next' in dependencies:
                        self.project_type = 'nextjs'
                    elif 'react' in dependencies:
                        self.project_type = 'react'
                    elif 'vue' in dependencies:
                        self.project_type = 'vue'
                    elif 'express' in dependencies:
                        self.project_type = 'express'
                    else:
                        self.project_type = 'nodejs'
                        
                    # Store dependencies
                    self.dependencies = set(dependencies.keys())
            except Exception:
                self.project_type = 'nodejs'
                
        # Check for Python projects
        elif any(f.endswith('.py') for f in files):
            if 'requirements.txt' in files:
                try:
                    with open(self.project_dir / 'requirements.txt', 'r') as f:
                        requirements = f.read().splitlines()
                        self.dependencies = set(r.split('==')[0].strip() for r in requirements if r.strip())
                        
                        # Check for specific frameworks
                        if 'django' in self.dependencies:
                            self.project_type = 'django'
                        elif 'flask' in self.dependencies:
                            self.project_type = 'flask'
                        elif 'fastapi' in self.dependencies:
                            self.project_type = 'fastapi'
                        else:
                            self.project_type = 'python'
                except Exception:
                    self.project_type = 'python'
            else:
                self.project_type = 'python'
                
        # Default to generic
        else:
            self.project_type = 'generic'
            
    def add_file(self, rel_path: str, content: str):
        """Add or update a file in the codebase."""
        file_path = self.project_dir / rel_path
        
        # Create directories if needed
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(file_path, 'w') as f:
            f.write(content)
            
        # Update internal tracking
        self.files[rel_path] = {
            'path': rel_path,
            'last_modified': file_path.stat().st_mtime,
            'size': file_path.stat().st_size,
            'content_hash': self._calculate_hash(file_path),
            'type': self._determine_file_type(file_path),
        }
        
        # Track if this was a creation or modification
        if rel_path in self.modified_files:
            self.modified_files.add(rel_path)
        else:
            self.created_files.add(rel_path)
            
        # If this is a special file that could change project type, re-detect
        if rel_path in ['package.json', 'requirements.txt']:
            self._detect_project_type()
            
        return file_path
        
    def get_file_content(self, rel_path: str) -> str:
        """Get the content of a file."""
        file_path = self.project_dir / rel_path
        if not file_path.exists():
            return ""
            
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception:
            return ""
            
    def get_project_info(self) -> dict:
        """Get information about the project."""
        return {
            'project_type': self.project_type,
            'file_count': len(self.files),
            'dependencies': list(self.dependencies),
            'created_files': list(self.created_files),
            'modified_files': list(self.modified_files),
        }
        
    def get_file_list(self, file_type=None) -> list:
        """Get a list of files, optionally filtered by type."""
        if file_type:
            return [info for path, info in self.files.items() if info['type'] == file_type]
        return list(self.files.values())
        
    def search_codebase(self, query: str) -> list:
        """Search the codebase for a specific query."""
        results = []
        
        for rel_path, info in self.files.items():
            try:
                file_path = self.project_dir / rel_path
                with open(file_path, 'r') as f:
                    content = f.read()
                    if query.lower() in content.lower():
                        results.append({
                            'path': rel_path,
                            'matches': content.lower().count(query.lower()),
                            'type': info['type']
                        })
            except Exception:
                continue
                
        # Sort by most matches
        results.sort(key=lambda x: x['matches'], reverse=True)
        return results
        
    def generate_project_summary(self) -> str:
        """Generate a summary of the project structure and files."""
        summary = []
        summary.append(f"Project Type: {self.project_type}")
        summary.append(f"Total Files: {len(self.files)}")
        
        if self.dependencies:
            summary.append(f"Dependencies: {', '.join(sorted(self.dependencies))}")
            
        # Group files by type
        files_by_type = {}
        for info in self.files.values():
            file_type = info['type']
            if file_type not in files_by_type:
                files_by_type[file_type] = []
            files_by_type[file_type].append(info['path'])
            
        for file_type, paths in files_by_type.items():
            summary.append(f"\n{file_type.capitalize()} files ({len(paths)}):")
            for path in sorted(paths)[:10]:  # Show at most 10 files per type
                summary.append(f"  - {path}")
            if len(paths) > 10:
                summary.append(f"  ... and {len(paths) - 10} more")
                
        return "\n".join(summary)

class AICLI:
    """CLI interface for AI coding agent with OpenAI integration."""
    
    def __init__(self):
        self.analyzer = CodeAnalyzer()
        self.reviewer = CodeReviewer()
        self.optimizer = CodeOptimizer()
        self.docgen = DocGenerator()
        self.codegen = CodeGenerator()
        self.conversation_history = []
        self.current_project = None
        self.project_manager = ProjectManager(Path("projects"))
        self.codebase_manager = None  # Will be initialized when project is created/selected
        self.running_processes = set()
    
    async def _process_response(self, response: str):
        """Process the AI response and handle code generation."""
        try:
            # Create a new project if none exists
            if not self.current_project:
                project_name = f"Project_{len(self.project_manager.templates) + 1}"
                self.current_project = self.project_manager.create_project("custom", project_name)
                print(f"\nCreated new project: {project_name}")
                # Initialize the codebase manager
                self.codebase_manager = CodebaseManager(self.current_project)
            elif self.codebase_manager is None:
                # Initialize codebase manager if not already done
                self.codebase_manager = CodebaseManager(self.current_project)
            
            # Pass the response to the code generator for processing
            # Include project info in the context for better context-aware code generation
            project_info = self.codebase_manager.get_project_info() if self.codebase_manager else {}
            context = {
                'ai_response': response,
                'requirements': "Process AI response and save code files",
                'project_info': project_info,
                'project_type': project_info.get('project_type', 'generic'),
                'dependencies': project_info.get('dependencies', []),
                'file_list': self.codebase_manager.get_file_list() if self.codebase_manager else []
            }
            
            result = await self.codegen.arun(context)
            
            # Process the generated code
            await self._process_generated_code(result)
            
            # If the code generator didn't find files, fall back to the old method
            if not result.get('code_files'):
                # Extract code blocks
                code_blocks = self._extract_code_blocks(response)
                if code_blocks:
                    await self._process_code_blocks(code_blocks)
            
            # After processing, update the user with a summary of what changed
            if self.codebase_manager:
                created = self.codebase_manager.created_files
                modified = self.codebase_manager.modified_files
                if created or modified:
                    print("\nChanges to the codebase:")
                    if created:
                        print(f"Created {len(created)} files: {', '.join(sorted(list(created)[:5]))}")
                        if len(created) > 5:
                            print(f"... and {len(created) - 5} more")
                    if modified:
                        print(f"Modified {len(modified)} files: {', '.join(sorted(list(modified)[:5]))}")
                        if len(modified) > 5:
                            print(f"... and {len(modified) - 5} more")
            
            # Print the response
            print("\n" + response)
            
        except KeyboardInterrupt:
            print("\nOperation interrupted. Saving current progress...")
            # Try to continue with what we have so far
            await self._handle_interrupted_operation(response)
        except Exception as e:
            logger.error(f"Error processing response: {str(e)}")
            print(f"Error processing response: {str(e)}")
            # Try recovery
            await self._handle_error_recovery(response)
    
    async def _process_generated_code(self, result):
        """Process code generated by the CodeGenerator."""
        # Check for code files
        if 'code_files' in result and result['code_files']:
            await self._process_code_files(result['code_files'])
            
            # Process bash files
            bash_files = [f for f in result['code_files'] if f.get('language') in ['bash', 'sh']]
            if bash_files:
                print("\nExecuting commands:")
                for bash_file in bash_files:
                    commands = [cmd.strip() for cmd in bash_file['content'].split('\n') if cmd.strip()]
                    for cmd in commands:
                        if cmd.startswith('#') or not cmd:  # Skip comments and empty lines
                            continue
                        print(f"\nExecuting: {cmd}")
                        await self._execute_command(cmd)
                        await asyncio.sleep(1)  # Small delay between commands
    
    async def _process_code_files(self, code_files):
        """Process and save code files."""
        if not code_files:
            return
            
        print(f"\nSaving {len(code_files)} files:")
        for file_info in code_files:
            try:
                file_path = file_info.get('path')
                if not file_path:
                    continue
                
                content = file_info.get('content', '')
                
                # Use the codebase manager to add the file if available
                if self.codebase_manager:
                    saved_path = self.codebase_manager.add_file(file_path, content)
                    print(f"  - {file_path} ({file_info.get('language', 'unknown')})")
                else:
                    # Make sure the path is within the project directory
                    if not file_path.startswith(str(self.current_project)):
                        absolute_path = os.path.join(str(self.current_project), file_path)
                    else:
                        absolute_path = file_path
                        
                    # Create directory structure if needed
                    dir_path = os.path.dirname(absolute_path)
                    if dir_path and not os.path.exists(dir_path):
                        os.makedirs(dir_path, exist_ok=True)
                        
                    # Write file content
                    with open(absolute_path, 'w') as f:
                        f.write(content)
                        
                    print(f"  - {os.path.relpath(absolute_path, str(self.current_project))} ({file_info.get('language', 'unknown')})")
                    
                    # Add to project metadata
                    rel_path = os.path.relpath(absolute_path, str(self.current_project))
                    self.project_manager.add_file(
                        self.current_project, 
                        rel_path, 
                        content
                    )
                    
            except Exception as e:
                logger.error(f"Error saving file {file_info.get('path')}: {str(e)}")
                print(f"Error saving file {file_info.get('path')}: {str(e)}")
    
    async def _process_code_blocks(self, code_blocks):
        """Process extracted code blocks."""
        try:
            # Load existing metadata
            metadata_path = self.current_project / "metadata.yaml"
            with open(metadata_path, "r") as f:
                metadata = yaml.safe_load(f)
            
            # Save each code block
            for i, (lang, code) in enumerate(code_blocks):
                # Determine file name based on content
                if "package.json" in code:
                    filename = "package.json"
                elif "server.js" in code or "app.js" in code:
                    filename = "server.js" if "server.js" in code else "app.js"
                elif "index.html" in code:
                    filename = "public/index.html"
                elif "App.js" in code or "App.jsx" in code:
                    filename = "src/App.js"
                elif "supabaseClient.js" in code:
                    filename = "lib/supabaseClient.js"
                elif ".env.local" in code:
                    filename = ".env.local"
                elif "main.py" in code and "FastAPI" in code:
                    filename = "main.py"
                else:
                    # Generate a unique filename
                    file_count = len([f for f in metadata["files"] if f["name"].startswith(f"file_{lang}")])
                    filename = f"file_{lang}_{file_count + 1}.{lang}"
                
                # Create necessary directories
                file_path = self.current_project / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Save the file
                self.project_manager.add_file(self.current_project, filename, code)
                print(f"\nSaved {filename}")
                
                # If it's a bash command, execute it
                if lang == "bash":
                    # Split commands by newline and filter out empty lines
                    commands = [cmd.strip() for cmd in code.split("\n") if cmd.strip()]
                    
                    # For each command, check if it's creating a Next.js app
                    for cmd in commands:
                        try:
                            print(f"\nExecuting: {cmd}")
                            
                            # Special handling for create-next-app
                            if "create-next-app" in cmd:
                                # Extract project name
                                project_match = re.search(r'create-next-app\s+(\S+)', cmd)
                                if project_match:
                                    project_dir = project_match.group(1)
                                    
                                    # Create project directory structure instead of running the command
                                    project_path = self.current_project / project_dir
                                    project_path.mkdir(parents=True, exist_ok=True)
                                    
                                    # Set up basic structure for Next.js
                                    (project_path / "pages").mkdir(exist_ok=True)
                                    (project_path / "styles").mkdir(exist_ok=True)
                                    (project_path / "public").mkdir(exist_ok=True)
                                    (project_path / "components").mkdir(exist_ok=True)
                                    
                                    # Create minimal package.json
                                    with open(project_path / "package.json", "w") as f:
                                        f.write('''
{
  "name": "my-supabase-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^12.0.0",
    "react": "^17.0.2",
    "react-dom": "^17.0.2",
    "@supabase/supabase-js": "^1.35.0"
  }
}
''')
                                    print(f"Created basic Next.js project structure in {project_dir}")
                                    
                                    # Update current directory
                                    self.current_project = project_path
                                    
                                    # No need to actually run the command
                                    continue
                            
                            # For other commands, use normal execution
                            await self._execute_command(cmd)
                            await asyncio.sleep(1)  # Small delay between commands
                        except KeyboardInterrupt:
                            print("\nCommand execution interrupted. Moving to next command...")
                            continue
                        except Exception as e:
                            print(f"Error executing command: {str(e)}. Moving to next command...")
                            continue
        except Exception as e:
            logger.error(f"Error processing code blocks: {str(e)}")
            print(f"Error processing code blocks: {str(e)}")
            
    async def _handle_interrupted_operation(self, response):
        """Handle interrupted operations and try to recover."""
        print("Attempting to salvage progress after interruption...")
        
        # Extract any code blocks that were interrupted
        try:
            code_blocks = self._extract_code_blocks(response)
            if code_blocks:
                print(f"Found {len(code_blocks)} code blocks to process.")
                # Process only the file saving part, skip execution
                for i, (lang, code) in enumerate(code_blocks):
                    try:
                        if lang != "bash":  # Skip bash execution on recovery
                            # Determine simple filename
                            ext = lang if lang != "javascript" else "js"
                            filename = f"recovered_file_{i}.{ext}"
                            
                            # Save the file
                            file_path = self.current_project / filename
                            with open(file_path, "w") as f:
                                f.write(code)
                                
                            print(f"Recovered: {filename}")
                    except Exception as inner_e:
                        print(f"Could not recover file {i}: {str(inner_e)}")
                        continue
        except Exception as e:
            print(f"Recovery attempt after interruption failed: {str(e)}")
            
        print("Recovery process completed. Continue with your next request.")
            
    async def _handle_error_recovery(self, response):
        """Attempt to recover from errors during processing."""
        print("Attempting to recover from errors...")
        
        try:
            # Create fallback files based on the general request
            if self.current_project:
                # Check if we can infer project type from the response
                if "next" in response.lower() or "react" in response.lower():
                    # Create a basic Next.js structure
                    await self._create_basic_nextjs_structure()
                elif "fast" in response.lower() and "api" in response.lower():
                    # Create a basic FastAPI structure
                    self._create_basic_fastapi_app()
                else:
                    # Create generic project files
                    self._create_mock_project_files()
                    
                print("Created basic project structure as fallback.")
        except Exception as e:
            print(f"Recovery attempt failed: {str(e)}")
            
    async def _create_basic_nextjs_structure(self):
        """Create a basic Next.js project structure as fallback."""
        if not self.current_project:
            return
            
        # Create directories
        for dir_name in ['pages', 'styles', 'public', 'components', 'lib']:
            os.makedirs(os.path.join(str(self.current_project), dir_name), exist_ok=True)
            
        # Create basic package.json
        with open(os.path.join(str(self.current_project), 'package.json'), 'w') as f:
            f.write('''{
  "name": "nextjs-app",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start"
  },
  "dependencies": {
    "next": "^12.0.0",
    "react": "^17.0.2",
    "react-dom": "^17.0.2"
  }
}''')
            
        # Create index.js
        with open(os.path.join(str(self.current_project), 'pages', 'index.js'), 'w') as f:
            f.write('''export default function Home() {
  return (
    <div>
      <h1>Welcome to Next.js!</h1>
      <p>This is a basic Next.js app.</p>
    </div>
  )
}''')
            
        # Create _app.js
        with open(os.path.join(str(self.current_project), 'pages', '_app.js'), 'w') as f:
            f.write('''import '../styles/globals.css'

export default function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}''')
            
        # Create globals.css
        with open(os.path.join(str(self.current_project), 'styles', 'globals.css'), 'w') as f:
            f.write('''html,
body {
  padding: 0;
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen,
    Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;
}

* {
  box-sizing: border-box;
}''')
            
        print("Created basic Next.js project structure")
        
    def _extract_code_blocks(self, response: str) -> List[tuple]:
        """Extract code blocks from the response."""
        code_blocks = []
        current_block = None
        current_lang = None
        
        for line in response.split("\n"):
            if line.startswith("```"):
                if current_block is None:
                    # Start of a code block
                    current_lang = line[3:].strip()
                    current_block = []
                else:
                    # End of a code block
                    if current_block:  # Only add non-empty blocks
                        code_blocks.append((current_lang, "\n".join(current_block)))
                    current_block = None
                    current_lang = None
            elif current_block is not None:
                current_block.append(line)
                
        return code_blocks
    
    async def _execute_command(self, command: str):
        """Execute a shell command with proper error handling and recovery."""
        try:
            # Split the command into parts
            parts = shlex.split(command)
            if not parts:
                return

            # Handle specific command patterns
            if '<filename>' in command:
                # Replace <filename> with an actual file
                possible_files = [f for f in os.listdir() if f.endswith('.py')]
                if possible_files:
                    command = command.replace('<filename>', os.path.splitext(possible_files[0])[0])
                    parts = shlex.split(command)
                else:
                    print("Error: No Python files found to replace <filename>")
                    return

            # Handle special commands
            if parts[0] == 'npx' and 'create-next-app' in command:
                print(f"Detected Next.js app creation command: {command}")
                # Extract project name
                project_match = re.search(r'create-next-app\S*\s+(\S+)', command)
                if project_match:
                    project_name = project_match.group(1)
                    print(f"Setting up Next.js project structure for: {project_name}")
                    
                    # Create project directory structure instead of running the command
                    project_dir = os.path.join(os.getcwd(), project_name)
                    os.makedirs(project_dir, exist_ok=True)
                    os.makedirs(os.path.join(project_dir, 'pages'), exist_ok=True)
                    os.makedirs(os.path.join(project_dir, 'styles'), exist_ok=True)
                    os.makedirs(os.path.join(project_dir, 'public'), exist_ok=True)
                    os.makedirs(os.path.join(project_dir, 'components'), exist_ok=True)
                    
                    # Create basic files
                    with open(os.path.join(project_dir, 'package.json'), 'w') as f:
                        f.write('''{
  "name": "''' + project_name + '''",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  },
  "dependencies": {
    "next": "^12.3.1",
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "eslint": "^8.25.0",
    "eslint-config-next": "^12.3.1"
  }
}''')
                    
                    with open(os.path.join(project_dir, 'pages', 'index.js'), 'w') as f:
                        f.write('''export default function Home() {
  return (
    <div>
      <h1>Welcome to Next.js!</h1>
      <p>Your Next.js app is ready.</p>
    </div>
  )
}''')
                    
                    with open(os.path.join(project_dir, 'pages', '_app.js'), 'w') as f:
                        f.write('''import '../styles/globals.css'

export default function MyApp({ Component, pageProps }) {
  return <Component {...pageProps} />
}''')
                    
                    with open(os.path.join(project_dir, 'styles', 'globals.css'), 'w') as f:
                        f.write('''html,
body {
  padding: 0;
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen,
    Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif;
}

a {
  color: inherit;
  text-decoration: none;
}

* {
  box-sizing: border-box;
}''')
                    
                    print(f"Created Next.js project structure in {project_name}")
                    self.current_project = Path(project_dir)
                    return
            
            # Handle development server launches
            if command.startswith('npm run dev') or 'uvicorn' in command:
                print(f"Detected server start command: {command}")
                print("Server would start here. Continuing with other operations...")
                return

            # Create the process with proper environment and working directory
            env = os.environ.copy()
            if self.current_project:
                env["PROJECT_DIR"] = str(self.current_project)
                cwd = str(self.current_project)
            else:
                cwd = None

            # Create the process
            print(f"Executing command: {command}")
            process = await asyncio.create_subprocess_exec(
                *parts,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=env
            )
            
            # Add to running processes
            self.running_processes.add(process)
            
            try:
                # Wait for the process to complete with a timeout
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30)
                
                # Print output
                if stdout:
                    print(stdout.decode())
                if stderr:
                    stderr_output = stderr.decode()
                    print(stderr_output)
                    
                    # Try to recover from specific errors
                    if "command not found" in stderr_output:
                        if parts[0] in ['npm', 'npx', 'node']:
                            print("Node.js might not be installed. Creating mock files instead.")
                            self._create_mock_project_files()
                    
                    elif "ENOENT" in stderr_output and "package.json" in stderr_output:
                        print("package.json not found. Creating basic package.json file.")
                        self._create_basic_package_json()
                    
                # Check return code
                if process.returncode != 0:
                    print(f"Command failed with return code {process.returncode}")
                    # Try to recover based on the command that failed
                    await self._recover_from_command_failure(command, process.returncode, stderr.decode() if stderr else "")
                    
            except asyncio.TimeoutError:
                print(f"Command timed out: {command}")
                process.terminate()
                await process.wait()
                
            finally:
                # Remove from running processes
                self.running_processes.discard(process)
                
        except Exception as e:
            logger.error(f"Command execution error: {str(e)}")
            print(f"Error executing command: {str(e)}")
            # Try to recover from the exception
            await self._recover_from_exception(command, str(e))

    def _create_mock_project_files(self):
        """Create basic project files when npm/node commands fail."""
        if not self.current_project:
            return
            
        # Create basic project structure
        print("Creating basic project structure...")
        os.makedirs(os.path.join(str(self.current_project), 'src'), exist_ok=True)
        
        # Create a basic index.js file
        with open(os.path.join(str(self.current_project), 'src', 'index.js'), 'w') as f:
            f.write('console.log("Hello, world!");')
            
        # Create a basic package.json
        self._create_basic_package_json()
        
        print("Created basic project files as a fallback.")
        
    def _create_basic_package_json(self):
        """Create a basic package.json file."""
        if not self.current_project:
            return
            
        package_path = os.path.join(str(self.current_project), 'package.json')
        
        with open(package_path, 'w') as f:
            f.write('''{
  "name": "project",
  "version": "1.0.0",
  "description": "A generated project",
  "main": "index.js",
  "scripts": {
    "start": "node src/index.js",
    "dev": "node src/index.js",
    "test": "echo \\"Error: no test specified\\" && exit 1"
  },
  "dependencies": {},
  "devDependencies": {}
}''')
        
        print(f"Created basic package.json at {package_path}")
        
    async def _recover_from_command_failure(self, command: str, return_code: int, stderr: str):
        """Try to recover from a failed command."""
        parts = shlex.split(command)
        
        if not parts:
            return
            
        # Handle specific command failures
        if parts[0] == 'npm' and 'install' in command:
            print("NPM install failed. Creating package.json with required dependencies.")
            # Extract package names from the command
            package_args = [p for p in parts if not p.startswith('-')]
            packages = package_args[2:] if len(package_args) > 2 else []
            
            if packages:
                self._add_dependencies_to_package_json(packages)
        
        elif parts[0] == 'uvicorn' and len(parts) > 1:
            # Handle uvicorn errors
            if "No module named" in stderr:
                print("Creating a basic FastAPI app file.")
                self._create_basic_fastapi_app()
                
    def _add_dependencies_to_package_json(self, packages: List[str]):
        """Add dependencies to package.json."""
        if not self.current_project:
            return
            
        package_path = os.path.join(str(self.current_project), 'package.json')
        
        # Create or update package.json
        if os.path.exists(package_path):
            try:
                with open(package_path, 'r') as f:
                    package_data = json.load(f)
            except json.JSONDecodeError:
                package_data = {
                    "name": "project",
                    "version": "1.0.0",
                    "dependencies": {},
                    "devDependencies": {}
                }
        else:
            package_data = {
                "name": "project",
                "version": "1.0.0",
                "dependencies": {},
                "devDependencies": {}
            }
            
        # Add dependencies
        if "dependencies" not in package_data:
            package_data["dependencies"] = {}
            
        for package in packages:
            if '@' in package and package[0] != '@':
                # Handle version specified packages like express@4.17.1
                name, version = package.split('@', 1)
                package_data["dependencies"][name] = f"^{version}"
            else:
                # For regular packages or scoped packages
                package_data["dependencies"][package] = "latest"
                
        # Write back to package.json
        with open(package_path, 'w') as f:
            json.dump(package_data, f, indent=2)
            
        print(f"Updated package.json with dependencies: {', '.join(packages)}")
        
    def _create_basic_fastapi_app(self):
        """Create a basic FastAPI application file."""
        if not self.current_project:
            return
            
        # Create the main.py file
        main_py_path = os.path.join(str(self.current_project), 'main.py')
        
        with open(main_py_path, 'w') as f:
            f.write('''from fastapi import FastAPI
from pydantic import BaseModel
import os

app = FastAPI(title="FastAPI App")

class Message(BaseModel):
    content: str

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI!"}

@app.post("/chat")
async def chat(message: Message):
    # This would typically call an OpenAI API
    return {"response": f"You said: {message.content}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
''')
        
        print(f"Created basic FastAPI app at {main_py_path}")
        print("You can now start the server with: uvicorn main:app --reload")
        
    async def _recover_from_exception(self, command: str, error_msg: str):
        """Try to recover from an exception during command execution."""
        parts = shlex.split(command)
        
        if not parts:
            return
            
        if "No such file or directory" in error_msg:
            # Handle missing file/directory errors
            if parts[0] in ['cd', 'mkdir', 'touch']:
                try:
                    # Create the directory or file
                    target = parts[1] if len(parts) > 1 else ""
                    if target:
                        if parts[0] == 'cd' or parts[0] == 'mkdir':
                            os.makedirs(target, exist_ok=True)
                            print(f"Created directory: {target}")
                        elif parts[0] == 'touch':
                            with open(target, 'w') as f:
                                pass
                            print(f"Created empty file: {target}")
                except Exception as e:
                    print(f"Recovery attempt failed: {str(e)}")
            
    async def cleanup(self):
        """Cleanup any running processes."""
        for process in self.running_processes:
            try:
                process.terminate()
                await process.wait()
            except Exception as e:
                logger.error(f"Error cleaning up process: {str(e)}")
        self.running_processes.clear()

    async def start(self):
        """Start the CLI interface."""
        print("Welcome to AI Coding Assistant!")
        print("Type 'exit' to quit, 'help' for commands")
        
        try:
            while True:
                try:
                    user_input = input("\nWhat would you like to create? > ")
                    
                    if user_input.lower() == 'exit':
                        break
                    elif user_input.lower() == 'help':
                        self._show_help()
                        continue
                    elif user_input.lower() == 'clear':
                        self.conversation_history = []
                        print("Conversation history cleared.")
                        continue
                    elif user_input.lower() == 'projects':
                        self._list_projects()
                        continue
                    elif user_input.lower().startswith('switch '):
                        project_id = user_input[7:].strip()
                        await self._switch_project(project_id)
                        continue
                    elif user_input.lower() == 'info':
                        self._show_project_info()
                        continue
                    elif user_input.lower().startswith('search '):
                        query = user_input[7:].strip()
                        self._search_codebase(query)
                        continue
                    elif user_input.lower().startswith('list '):
                        file_type = user_input[5:].strip()
                        self._list_files(file_type)
                        continue
                    
                    # Add user input to conversation history
                    self.conversation_history.append({"role": "user", "content": user_input})
                    
                    # Get AI response
                    response = await self._get_ai_response(user_input)
                    
                    # Add AI response to conversation history
                    self.conversation_history.append({"role": "assistant", "content": response})
                    
                    # Process the response
                    await self._process_response(response)
                    
                except KeyboardInterrupt:
                    print("\nInterrupted. Type 'exit' to quit or continue with your request.")
                    continue
                except Exception as e:
                    logger.error(f"Error: {str(e)}")
                    print(f"An error occurred: {str(e)}")
                    
        finally:
            await self.cleanup()

    def _list_projects(self):
        """List all available projects."""
        if not self.project_manager.templates:
            print("\nNo projects found.")
            return
            
        print("\nAvailable projects:")
        for project_name, metadata in self.project_manager.templates.items():
            status = " (current)" if project_name == self.current_project.name else ""
            print(f"  {project_name} - {metadata['name']}{status}")
            print(f"    Created: {metadata['created_at']}")
            print(f"    Last modified: {metadata['last_modified']}")
            print(f"    Description: {metadata['description']}")
            print()
    
    async def _switch_project(self, project_id):
        """Switch to a different project."""
        # Check if it's a UUID (from the project directory)
        try:
            project_path = Path("projects") / project_id
            if project_path.exists() and project_path.is_dir():
                self.current_project = project_path
                # Reset codebase manager for the new project
                self.codebase_manager = CodebaseManager(self.current_project)
                print(f"\nSwitched to project: {project_id}")
                # Show project info
                self._show_project_info()
                return
        except Exception:
            pass
            
        # Otherwise try to find it by name
        if project_id not in self.project_manager.templates:
            print(f"\nProject {project_id} not found.")
            return
            
        self.current_project = self.project_manager.create_project("custom", project_id)
        # Reset codebase manager for the new project
        self.codebase_manager = CodebaseManager(self.current_project)
        print(f"\nSwitched to project: {project_id}")
        # Show project info
        self._show_project_info()
        
    def _show_project_info(self):
        """Show information about the current project."""
        if not self.current_project:
            print("\nNo project selected.")
            return
            
        if not self.codebase_manager:
            self.codebase_manager = CodebaseManager(self.current_project)
            
        print("\nProject information:")
        print(self.codebase_manager.generate_project_summary())
    
    def _search_codebase(self, query):
        """Search the codebase for a query."""
        if not self.current_project:
            print("\nNo project selected.")
            return
            
        if not self.codebase_manager:
            self.codebase_manager = CodebaseManager(self.current_project)
            
        results = self.codebase_manager.search_codebase(query)
        
        if not results:
            print(f"\nNo results found for query: '{query}'")
            return
            
        print(f"\nSearch results for '{query}':")
        for result in results[:10]:  # Show top 10 results
            print(f"  - {result['path']} ({result['matches']} matches)")
            
        if len(results) > 10:
            print(f"  ... and {len(results) - 10} more matches")
    
    def _list_files(self, file_type=None):
        """List files in the current project."""
        if not self.current_project:
            print("\nNo project selected.")
            return
            
        if not self.codebase_manager:
            self.codebase_manager = CodebaseManager(self.current_project)
            
        if file_type and file_type != 'all':
            files = self.codebase_manager.get_file_list(file_type)
            print(f"\n{file_type.capitalize()} files in the project:")
        else:
            files = self.codebase_manager.get_file_list()
            print("\nAll files in the project:")
            
        if not files:
            print("  No files found.")
            return
            
        for file in sorted(files, key=lambda x: x['path'])[:20]:  # Show up to 20 files
            print(f"  - {file['path']} ({file['type']})")
            
        if len(files) > 20:
            print(f"  ... and {len(files) - 20} more files")
    
    async def _get_ai_response(self, user_input: str) -> str:
        """Get response from OpenAI with web search integration and codebase context."""
        try:
            # Check if the input is requesting web knowledge
            needs_web_info = any(keyword in user_input.lower() for keyword in ['create', 'setup', 'install', 'build', 'latest', 'how to', 'tutorial'])
            
            # Gather codebase context
            codebase_context = ""
            if self.codebase_manager:
                project_info = self.codebase_manager.get_project_info()
                
                # Add basic project info
                codebase_context += f"Project Type: {project_info['project_type']}\n"
                codebase_context += f"Total Files: {project_info['file_count']}\n"
                
                # Add dependencies
                if project_info['dependencies']:
                    codebase_context += f"Dependencies: {', '.join(project_info['dependencies'][:10])}"
                    if len(project_info['dependencies']) > 10:
                        codebase_context += f" and {len(project_info['dependencies']) - 10} more"
                    codebase_context += "\n"
                
                # Add recently modified/created files
                if project_info['modified_files']:
                    codebase_context += f"Recently Modified Files: {', '.join(list(project_info['modified_files'])[:5])}"
                    if len(project_info['modified_files']) > 5:
                        codebase_context += f" and {len(project_info['modified_files']) - 5} more"
                    codebase_context += "\n"
                
                # Try to find relevant files based on the user query
                search_results = self.codebase_manager.search_codebase(user_input)
                if search_results:
                    codebase_context += f"\nRelevant Files for Your Query:\n"
                    for result in search_results[:5]:
                        file_content = self.codebase_manager.get_file_content(result['path'])
                        # Add truncated content if not too large
                        if len(file_content) < 1000:
                            codebase_context += f"--- {result['path']} ---\n{file_content}\n\n"
                        else:
                            # Just add a summary
                            codebase_context += f"--- {result['path']} (summarized) ---\n{file_content[:500]}...\n\n"
            
            if needs_web_info:
                # Get web information first
                search_results = await self._execute_web_search(user_input)
                
                # Prepare the conversation with web search results and codebase context
                system_message = """You are an AI coding assistant. Help users create and modify code.
                Use the provided web search information where relevant to ensure you're giving up-to-date guidance.
                
                CODEBASE CONTEXT:
                {codebase_context}
                
                If asked to create projects or setup development environments, include step-by-step instructions with:
                1. Prerequisites installation commands
                2. Project initialization commands
                3. Required file structure with code examples
                4. Configuration files and environment variables
                5. Running/testing commands
                
                Always consider the existing codebase structure and dependencies when suggesting changes or new code.
                """
                
                messages = [
                    {"role": "system", "content": system_message.format(codebase_context=codebase_context if codebase_context else "No existing codebase.")},
                    {"role": "user", "content": user_input},
                    {"role": "assistant", "content": f"Here's what I found from searching about your request: {search_results}"},
                    {"role": "user", "content": "Based on this information and the codebase context, please provide a detailed response to my original request, including code samples in markdown format with ```language blocks."}
                ]
            else:
                # Prepare the normal conversation with codebase context
                system_message = """You are an AI coding assistant. Help users create and modify code.
                
                CODEBASE CONTEXT:
                {codebase_context}
                
                Always consider the existing codebase structure and dependencies when suggesting changes or new code.
                When generating new files, ensure they fit with the existing project structure and use consistent naming conventions.
                """
                
                messages = [
                    {"role": "system", "content": system_message.format(codebase_context=codebase_context if codebase_context else "No existing codebase.")},
                ] + self.conversation_history[-5:]  # Keep last 5 messages for context
                
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.7,
                max_tokens=1500  # Increase token limit for more detailed responses
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"OpenAI API error: {str(e)}")
            return f"I encountered an error: {str(e)}"
            
    async def _execute_web_search(self, query: str) -> str:
        """Execute web search and return formatted results."""
        try:
            # Use the OpenAI API for search information, without temperature parameter
            search_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that searches the web for information. Provide only factual information based on reliable sources."},
                    {"role": "user", "content": f"Search for information about: {query}"}
                ],
                max_tokens=500
            )
            return search_response.choices[0].message.content
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return f"Error performing web search: {str(e)}"
    
    def _show_help(self):
        """Show available commands."""
        print("\nAvailable commands:")
        print("  help          - Show this help message")
        print("  exit          - Exit the CLI")
        print("  clear         - Clear conversation history")
        print("  projects      - List all projects")
        print("  switch <name> - Switch to a different project")
        print("  info          - Show information about the current project")
        print("  search <query> - Search the codebase for a specific query")
        print("  list [type]   - List files in the current project, optionally filtered by type")
        print("\nYou can also:")
        print("  - Ask for code generation")
        print("  - Request code modifications")
        print("  - Ask for explanations")
        print("  - Request code reviews")
        print("  - Ask for debugging help")

async def main():
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set your OpenAI API key as an environment variable:")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    # Initialize and start the CLI
    cli = AICLI()
    try:
        await cli.start()
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}")
    finally:
        await cli.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"Fatal error: {str(e)}") 