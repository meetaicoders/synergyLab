import asyncio
import re
import json
from typing import Dict, Any, List, Optional
import logging
from contextlib import asynccontextmanager
import shlex
import os

# Import the Agent framework
# Assuming paste.txt is saved as paste.py
from mertlesh_sonic import Agent, FunctionAgent, AgentFlow

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AICodingAgent")

class CodeAnalyzer(Agent):
    """Agent that analyzes code structure and identifies patterns."""
    
    def __init__(self, name: str = "analyzer"):
        super().__init__(name)
        
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context.get('code', '')
        if not code:
            context['analysis'] = {
                'status': 'error',
                'message': 'No code provided for analysis'
            }
            return context
            
        # Analyze the code structure
        analysis = {}
        
        # Identify language
        language = self._detect_language(code)
        analysis['language'] = language
        
        # Function/class detection
        if language == 'python':
            analysis['functions'] = self._extract_python_functions(code)
            analysis['classes'] = self._extract_python_classes(code)
            analysis['imports'] = self._extract_python_imports(code)
        elif language == 'javascript':
            analysis['functions'] = self._extract_js_functions(code)
            analysis['classes'] = self._extract_js_classes(code)
            analysis['imports'] = self._extract_js_imports(code)
            
        # Add analysis to context
        context['analysis'] = {
            'status': 'success',
            'result': analysis
        }
        
        return context
        
    def _detect_language(self, code: str) -> str:
        """Detect the programming language of the code."""
        # Simple heuristics for language detection
        if re.search(r'import\s+|from\s+\w+\s+import|def\s+\w+\s*\(|class\s+\w+:', code):
            return 'python'
        elif re.search(r'function\s+\w+\s*\(|const\s+|let\s+|var\s+|import\s+.+from', code):
            return 'javascript'
        elif re.search(r'#include|int\s+main\(|void\s+\w+\s*\(', code):
            return 'c/cpp'
        else:
            return 'unknown'
            
    def _extract_python_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function definitions from Python code."""
        functions = []
        pattern = r'def\s+(\w+)\s*\((.*?)\)(?:\s*->.*?)?:'
        
        for match in re.finditer(pattern, code, re.DOTALL):
            name = match.group(1)
            params = match.group(2).strip()
            functions.append({
                'name': name,
                'params': [p.strip().split('=')[0].strip() for p in params.split(',') if p.strip()]
            })
            
        return functions
        
    def _extract_python_classes(self, code: str) -> List[Dict[str, Any]]:
        """Extract class definitions from Python code."""
        classes = []
        pattern = r'class\s+(\w+)(?:\s*\((.+?)\))?:'
        
        for match in re.finditer(pattern, code):
            name = match.group(1)
            inheritance = match.group(2) if match.group(2) else ''
            classes.append({
                'name': name,
                'inherits': [c.strip() for c in inheritance.split(',')] if inheritance else []
            })
            
        return classes
        
    def _extract_python_imports(self, code: str) -> List[str]:
        """Extract imports from Python code."""
        imports = []
        import_pattern = r'import\s+([\w\.]+)'
        from_import_pattern = r'from\s+([\w\.]+)\s+import\s+(.+)'
        
        for match in re.finditer(import_pattern, code):
            imports.append(match.group(1))
            
        for match in re.finditer(from_import_pattern, code):
            module = match.group(1)
            items = match.group(2)
            imports.append(f"{module}: {items}")
            
        return imports
        
    def _extract_js_functions(self, code: str) -> List[Dict[str, Any]]:
        """Extract function definitions from JavaScript code."""
        functions = []
        # Regular functions
        pattern1 = r'function\s+(\w+)\s*\((.*?)\)'
        # Arrow functions with name
        pattern2 = r'(?:const|let|var)\s+(\w+)\s*=\s*(?:\(.*?\)|\w+)\s*=>'
        
        for match in re.finditer(pattern1, code, re.DOTALL):
            name = match.group(1)
            params = match.group(2).strip()
            functions.append({
                'name': name,
                'params': [p.strip().split('=')[0].strip() for p in params.split(',') if p.strip()]
            })
            
        for match in re.finditer(pattern2, code):
            name = match.group(1)
            functions.append({
                'name': name,
                'type': 'arrow'
            })
            
        return functions
        
    def _extract_js_classes(self, code: str) -> List[Dict[str, Any]]:
        """Extract class definitions from JavaScript code."""
        classes = []
        pattern = r'class\s+(\w+)(?:\s+extends\s+(\w+))?'
        
        for match in re.finditer(pattern, code):
            name = match.group(1)
            extends = match.group(2) if match.group(2) else None
            classes.append({
                'name': name,
                'extends': extends
            })
            
        return classes
        
    def _extract_js_imports(self, code: str) -> List[str]:
        """Extract imports from JavaScript code."""
        imports = []
        import_pattern = r'import\s+(?:{(.+?)}|(.+?))\s+from\s+[\'"](.+?)[\'"]'
        
        for match in re.finditer(import_pattern, code):
            named_imports = match.group(1)
            default_import = match.group(2)
            source = match.group(3)
            
            if named_imports:
                imports.append(f"{source}: {named_imports}")
            elif default_import:
                imports.append(f"{source}: {default_import}")
                
        return imports


class CodeReviewer(Agent):
    """Agent that reviews code and provides feedback."""
    
    def __init__(self, name: str = "reviewer"):
        super().__init__(name)
        
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context.get('code', '')
        analysis = context.get('analysis', {}).get('result', {})
        
        if not code or not analysis:
            context['review'] = {
                'status': 'error',
                'message': 'Missing code or analysis for review'
            }
            return context
            
        # Generate code review
        language = analysis.get('language', 'unknown')
        review_results = []
        
        # Check for common issues based on language
        if language == 'python':
            review_results.extend(self._review_python_code(code))
        elif language == 'javascript':
            review_results.extend(self._review_js_code(code))
            
        # General review checks for all languages
        review_results.extend(self._general_code_review(code))
        
        # Add review to context
        context['review'] = {
            'status': 'success',
            'issues': review_results,
            'summary': self._generate_review_summary(review_results)
        }
        
        return context
        
    def _review_python_code(self, code: str) -> List[Dict[str, Any]]:
        """Review Python-specific code issues."""
        issues = []
        
        # Check for bare exceptions
        if re.search(r'except:', code):
            issues.append({
                'type': 'warning',
                'message': 'Using bare except clause is not recommended',
                'suggestion': 'Specify exception types to catch'
            })
            
        # Check for mutable default arguments
        if re.search(r'def\s+\w+\s*\(.*?=\s*\[\].*?\)', code):
            issues.append({
                'type': 'warning',
                'message': 'Using mutable default argument (list)',
                'suggestion': 'Use None as default and initialize inside function'
            })
            
        # Check for unused imports
        import_matches = re.finditer(r'import\s+([\w\.]+)', code)
        for match in import_matches:
            imported = match.group(1)
            if not re.search(rf'[^\.\'"]({imported})[^\.\'"]*', code[match.end():]):
                issues.append({
                    'type': 'info',
                    'message': f'Potentially unused import: {imported}',
                    'suggestion': 'Remove if not used'
                })
                
        return issues
        
    def _review_js_code(self, code: str) -> List[Dict[str, Any]]:
        """Review JavaScript-specific code issues."""
        issues = []
        
        # Check for var usage
        if re.search(r'var\s+', code):
            issues.append({
                'type': 'suggestion',
                'message': 'Using var for variable declaration',
                'suggestion': 'Consider using const or let instead'
            })
            
        # Check for console.log
        if re.search(r'console\.log\(', code):
            issues.append({
                'type': 'info',
                'message': 'Found console.log statements',
                'suggestion': 'Remove debugging console.log statements before production'
            })
            
        # Check for potential memory leaks in event listeners
        if re.search(r'addEventListener\(', code) and not re.search(r'removeEventListener\(', code):
            issues.append({
                'type': 'warning',
                'message': 'Event listeners added without removal',
                'suggestion': 'Make sure to remove event listeners to prevent memory leaks'
            })
            
        return issues
        
    def _general_code_review(self, code: str) -> List[Dict[str, Any]]:
        """Review general code issues applicable to any language."""
        issues = []
        
        # Check for TODO comments
        todo_matches = re.finditer(r'(?:^|\s)(?:#|//)\s*TODO[:\s]*(.*?)(?:\n|$)', code, re.MULTILINE | re.IGNORECASE)
        for match in todo_matches:
            todo_text = match.group(1).strip()
            issues.append({
                'type': 'info',
                'message': f'TODO comment: {todo_text}',
                'suggestion': 'Consider addressing this TODO item'
            })
            
        # Check for long lines
        lines = code.split('\n')
        for i, line in enumerate(lines):
            if len(line) > 100:
                issues.append({
                    'type': 'style',
                    'line': i + 1,
                    'message': f'Line {i+1} is longer than 100 characters ({len(line)})',
                    'suggestion': 'Consider breaking this line into multiple lines'
                })
                
        # Check for trailing whitespace
        for i, line in enumerate(lines):
            if line.rstrip() != line:
                issues.append({
                    'type': 'style',
                    'line': i + 1,
                    'message': f'Line {i+1} has trailing whitespace',
                    'suggestion': 'Remove trailing whitespace'
                })
                
        return issues
        
    def _generate_review_summary(self, issues: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a summary of the code review issues."""
        counts = {
            'error': 0,
            'warning': 0,
            'info': 0,
            'style': 0,
            'suggestion': 0
        }
        
        for issue in issues:
            issue_type = issue.get('type', 'info')
            counts[issue_type] = counts.get(issue_type, 0) + 1
            
        total_issues = sum(counts.values())
        
        # Generate an overall code quality score (0-100)
        score = 100
        score -= counts['error'] * 10
        score -= counts['warning'] * 5
        score -= counts['style'] * 1
        score = max(0, score)
        
        return {
            'counts': counts,
            'total': total_issues,
            'quality_score': score,
            'verdict': self._get_verdict(score)
        }
        
    def _get_verdict(self, score: int) -> str:
        """Get a verdict based on the code quality score."""
        if score >= 90:
            return "Excellent code quality"
        elif score >= 80:
            return "Good code quality with minor issues"
        elif score >= 60:
            return "Acceptable code quality with some issues to address"
        elif score >= 40:
            return "Poor code quality with significant issues"
        else:
            return "Critical code quality issues that need immediate attention"


class CodeOptimizer(Agent):
    """Agent that suggests code optimizations."""
    
    def __init__(self, name: str = "optimizer"):
        super().__init__(name)
        
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context.get('code', '')
        analysis = context.get('analysis', {}).get('result', {})
        
        if not code or not analysis:
            context['optimizations'] = {
                'status': 'error',
                'message': 'Missing code or analysis for optimization'
            }
            return context
            
        # Generate optimization suggestions
        language = analysis.get('language', 'unknown')
        optimizations = []
        
        # Check for optimization opportunities based on language
        if language == 'python':
            optimizations.extend(self._optimize_python_code(code))
        elif language == 'javascript':
            optimizations.extend(self._optimize_js_code(code))
            
        # General optimization checks
        optimizations.extend(self._general_optimizations(code))
        
        # Add optimizations to context
        context['optimizations'] = {
            'status': 'success',
            'suggestions': optimizations
        }
        
        return context
        
    def _optimize_python_code(self, code: str) -> List[Dict[str, Any]]:
        """Find Python-specific optimization opportunities."""
        optimizations = []
        
        # Check for inefficient list comprehensions vs generator expressions
        if re.search(r'\[.+for.+in.+\]', code) and re.search(r'for.+in\s*\[.+for.+in.+\]', code):
            optimizations.append({
                'type': 'performance',
                'message': 'Nested list comprehension used where generator expression would be more efficient',
                'suggestion': 'Use generator expressions when the intermediate list is only used for iteration'
            })
            
        # Check for string concatenation in loops
        if re.search(r'for.+:.+\+=\s*[\'"]', code):
            optimizations.append({
                'type': 'performance',
                'message': 'String concatenation in loop',
                'suggestion': 'Use join() method or list comprehension instead of += for string concatenation in loops'
            })
            
        # Check for repeated dictionary access
        dict_accesses = re.findall(r'(\w+)\[[\'"]\w+[\'"]\]', code)
        for dict_name in set(dict_accesses):
            if dict_accesses.count(dict_name) > 3:
                optimizations.append({
                    'type': 'performance',
                    'message': f'Repeated dictionary access to {dict_name}',
                    'suggestion': f'Consider using a local variable to store frequently accessed values from {dict_name}'
                })
                
        return optimizations
        
    def _optimize_js_code(self, code: str) -> List[Dict[str, Any]]:
        """Find JavaScript-specific optimization opportunities."""
        optimizations = []
        
        # Check for DOM queries inside loops
        if re.search(r'for.+{.*document\.query', code, re.DOTALL):
            optimizations.append({
                'type': 'performance',
                'message': 'DOM query inside a loop',
                'suggestion': 'Cache DOM queries outside loops to avoid reflow and improve performance'
            })
            
        # Check for array push in a loop
        if re.search(r'for.+{.*\.push\(', code, re.DOTALL):
            optimizations.append({
                'type': 'performance',
                'message': 'Array.push() inside a loop',
                'suggestion': 'Consider pre-allocating array size or using a single map/filter operation'
            })
            
        # Check for +/+= string concatenation
        if re.search(r'[\'"]\s*\+|\+=\s*[\'"]', code):
            optimizations.append({
                'type': 'performance',
                'message': 'String concatenation with + operator',
                'suggestion': 'Consider using template literals for multi-part strings'
            })
            
        return optimizations
        
    def _general_optimizations(self, code: str) -> List[Dict[str, Any]]:
        """Find general optimization opportunities applicable to any language."""
        optimizations = []
        
        # Check for nested loops
        if re.search(r'for.+for.+:', code) or re.search(r'for.+for.+{', code):
            optimizations.append({
                'type': 'performance',
                'message': 'Nested loops detected',
                'suggestion': 'Consider if data structure or algorithm can be optimized to avoid O(n²) complexity'
            })
            
        # Check for repeated calculations
        lines = code.split('\n')
        repeated_expressions = set()
        
        for i in range(len(lines) - 1):
            for j in range(i + 1, len(lines)):
                # Look for complex calculations that are repeated
                complex_expr = re.search(r'=\s*(.+[\+\-\*\/].+)', lines[i])
                if complex_expr and complex_expr.group(1) in lines[j]:
                    repeated_expressions.add(complex_expr.group(1).strip())
                    
        for expr in repeated_expressions:
            if len(expr) > 10:  # Only suggest for non-trivial expressions
                optimizations.append({
                    'type': 'performance',
                    'message': f'Repeated calculation: "{expr}"',
                    'suggestion': 'Store the result in a variable to avoid recalculating'
                })
                
        return optimizations


class DocGenerator(Agent):
    """Agent that generates documentation for code."""
    
    def __init__(self, name: str = "docgen"):
        super().__init__(name)
        
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        code = context.get('code', '')
        analysis = context.get('analysis', {}).get('result', {})
        
        if not code or not analysis:
            context['documentation'] = {
                'status': 'error',
                'message': 'Missing code or analysis for documentation'
            }
            return context
            
        # Generate documentation
        language = analysis.get('language', 'unknown')
        docs = {}
        
        # Generate overall documentation
        docs['overview'] = self._generate_overview(code, analysis)
        
        # Generate documentation for functions and classes
        if language == 'python':
            docs['functions'] = self._document_python_functions(code, analysis)
            docs['classes'] = self._document_python_classes(code, analysis)
        elif language == 'javascript':
            docs['functions'] = self._document_js_functions(code, analysis)
            docs['classes'] = self._document_js_classes(code, analysis)
            
        # Add documentation to context
        context['documentation'] = {
            'status': 'success',
            'docs': docs,
            'markdown': self._generate_markdown_docs(docs, language)
        }
        
        return context
        
    def _generate_overview(self, code: str, analysis: Dict[str, Any]) -> str:
        """Generate an overview of the code."""
        language = analysis.get('language', 'unknown')
        functions_count = len(analysis.get('functions', []))
        classes_count = len(analysis.get('classes', []))
        
        lines_count = len(code.split('\n'))
        
        overview = f"This is a {language} code snippet consisting of {lines_count} lines. "
        
        if functions_count > 0:
            overview += f"It contains {functions_count} function"
            overview += "s" if functions_count > 1 else ""
            overview += ". "
            
        if classes_count > 0:
            overview += f"It defines {classes_count} class"
            overview += "es" if classes_count > 1 else ""
            overview += ". "
            
        # Try to infer the purpose of the code
        purpose = self._infer_code_purpose(code, analysis)
        if purpose:
            overview += f"The code appears to be {purpose}."
            
        return overview
        
    def _infer_code_purpose(self, code: str, analysis: Dict[str, Any]) -> str:
        """Infer the general purpose of the code."""
        # Check for certain patterns to guess the code's purpose
        if any(f.lower() in code.lower() for f in ['flask', 'django', 'fastapi', 'route', 'endpoint']):
            return "a web server or API"
        elif any(f.lower() in code.lower() for f in ['test_', 'assert', 'unittest', 'pytest']):
            return "a test suite"
        elif any(f.lower() in code.lower() for f in ['pandas', 'numpy', 'plt', 'matplotlib', 'seaborn']):
            return "data analysis or visualization code"
        elif any(f.lower() in code.lower() for f in ['train', 'model', 'predict', 'neural', 'classifier']):
            return "machine learning code"
        elif any(f.lower() in code.lower() for f in ['widget', 'button', 'window', 'render', 'component']):
            return "a user interface implementation"
        else:
            return "a utility or library code"
            
    def _document_python_functions(self, code: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document Python functions."""
        functions = analysis.get('functions', [])
        documented_functions = []
        
        for func in functions:
            name = func.get('name', '')
            params = func.get('params', [])
            
            # Extract function body to infer its purpose
            pattern = rf'def\s+{re.escape(name)}\s*\(.*?\):(.+?)(?=\n\S|\Z)'
            match = re.search(pattern, code, re.DOTALL)
            
            description = "No description available."
            returns = "Unknown return value."
            
            if match:
                body = match.group(1)
                
                # Check for existing docstring
                docstring_match = re.search(r'"""(.+?)"""', body, re.DOTALL)
                if docstring_match:
                    docstring = docstring_match.group(1).strip()
                    # Use existing docstring if available
                    description = docstring
                else:
                    # Try to infer purpose from function name and body
                    if name.startswith('get_') or name.startswith('fetch_'):
                        description = f"Retrieves {name[4:]} data."
                    elif name.startswith('set_') or name.startswith('update_'):
                        description = f"Updates the {name[4:]} value."
                    elif name.startswith('is_') or name.startswith('has_') or name.startswith('check_'):
                        description = f"Checks if {name[3:]} condition is met."
                    else:
                        # Generic description based on first line of code
                        first_line = body.strip().split('\n')[0].strip()
                        if first_line:
                            description = f"Function that performs operations related to {name}."
                
                # Try to determine return value
                if 'return' in body:
                    return_match = re.search(r'return\s+(.+?)(?:$|\n)', body)
                    if return_match:
                        returns = f"Returns {return_match.group(1).strip()}."
                        
            documented_functions.append({
                'name': name,
                'params': params,
                'description': description,
                'returns': returns
            })
            
        return documented_functions
        
    def _document_python_classes(self, code: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document Python classes."""
        classes = analysis.get('classes', [])
        documented_classes = []
        
        for cls in classes:
            name = cls.get('name', '')
            inherits = cls.get('inherits', [])
            
            # Extract class body to find methods and docstring
            pattern = rf'class\s+{re.escape(name)}(?:\(.+?\))?:(.+?)(?=\n(?:class|def\s+\w+(?!\s+self)|\Z))'
            match = re.search(pattern, code, re.DOTALL)
            
            description = "No description available."
            methods = []
            
            if match:
                body = match.group(1)
                
                # Check for existing docstring
                docstring_match = re.search(r'"""(.+?)"""', body, re.DOTALL)
                if docstring_match:
                    docstring = docstring_match.group(1).strip()
                    description = docstring
                    
                # Extract methods
                method_pattern = r'def\s+(\w+)\s*\(\s*self(?:,\s*(.+))?\s*\):'
                method_matches = re.finditer(method_pattern, body)
                
                for method_match in method_matches:
                    method_name = method_match.group(1)
                    method_params = method_match.group(2)
                    
                    params_list = []
                    if method_params:
                        params_list = [p.strip().split('=')[0].strip() for p in method_params.split(',')]
                        
                    # Skip private methods
                    if method_name.startswith('_') and method_name != '__init__':
                        continue
                        
                    methods.append({
                        'name': method_name,
                        'params': params_list
                    })
                    
            documented_classes.append({
                'name': name,
                'inherits': inherits,
                'description': description,
                'methods': methods
            })
            
        return documented_classes
        
    def _document_js_functions(self, code: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document JavaScript functions."""
        functions = analysis.get('functions', [])
        documented_functions = []
        
        for func in functions:
            name = func.get('name', '')
            params = func.get('params', []) if 'params' in func else []
            
            # Extract function body
            pattern = rf'function\s+{re.escape(name)}\s*\(.*?\)\s*{{(.+?)}}|const\s+{re.escape(name)}\s*=\s*(?:\(.*?\))?\s*=>\s*{{(.+?)}}'
            match = re.search(pattern, code, re.DOTALL)
            
            description = "No description available."
            returns = "Unknown return value."
            
            if match:
                body = match.group(1) or match.group(2) or ""
                
                # Check for JSDoc comment
                jsdoc_pattern = rf'/\*\*(.+?)\*/\s*(?:function\s+{re.escape(name)}|const\s+{re.escape(name)})'
                jsdoc_match = re.search(jsdoc_pattern, code, re.DOTALL)
                
                if jsdoc_match:
                    jsdoc = jsdoc_match.group(1).strip()
                    # Extract description from JSDoc
                    desc_match = re.search(r'^\s*\*\s+([^@].+?)(?=\n\s*\*\s*@|\n\s*\*/|$)', jsdoc, re.MULTILINE)
                    if desc_match:
                        description = desc_match.group(1).strip()
                        
                    # Extract return info from JSDoc
                    return_match = re.search(r'@returns?\s+(.+?)(?=\n\s*\*\s*@|\n\s*\*/|$)', jsdoc, re.MULTILINE)
                    if return_match:
                        returns = return_match.group(1).strip()
                else:
                    # Try to infer purpose from function name
                    if name.startswith('get') or name.startswith('fetch'):
                        description = f"Retrieves {name[3:]} data."
                    elif name.startswith('set') or name.startswith('update'):
                        description = f"Updates the {name[3:]} value."
                    elif name.startswith('is') or name.startswith('has'):
                        description = f"Checks if {name[2:]} condition is met."
                    else:
                        description = f"Function that performs operations related to {name}."
                        
                # Try to determine return value if not found in JSDoc
                if returns == "Unknown return value." and 'return' in body:
                    return_match = re.search(r'return\s+(.+?)(?:;|$|\n)', body)
                    if return_match:
                        returns = f"Returns {return_match.group(1).strip()}."
                        
            documented_functions.append({
                'name': name,
                'params': params,
                'description': description,
                'returns': returns
            })
            
        return documented_functions
        
    def _document_js_classes(self, code: str, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Document JavaScript classes."""
        classes = analysis.get('classes', [])
        documented_classes = []
        
        for cls in classes:
            name = cls.get('name', '')
            extends = cls.get('extends')
            
            # Extract class body
            pattern = rf'class\s+{re.escape(name)}(?:\s+extends\s+\w+)?\s*{{(.+?)}}'
            match = re.search(pattern, code, re.DOTALL)
            
            description = "No description available."
            methods = []
            
            if match:
                body = match.group(1)
                
                # Check for JSDoc comment
                jsdoc_pattern = rf'/\*\*(.+?)\*/\s*class\s+{re.escape(name)}'
                jsdoc_match = re.search(jsdoc_pattern, code, re.DOTALL)
                
                if jsdoc_match:
                    jsdoc = jsdoc_match.group(1).strip()
                    # Extract description from JSDoc
                    desc_match = re.search(r'^\s*\*\s+([^@].+?)(?=\n\s*\*\s*@|\n\s*\*/|$)', jsdoc, re.MULTILINE)
                    if desc_match:
                        description = desc_match.group(1).strip()
                
                # Extract methods
                method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(\w+)\s*\((.*?)\)\s*{'
                method_matches = re.finditer(method_pattern, body)
                
                for method_match in method_matches:
                    method_name = method_match.group(1)
                    method_params = method_match.group(2)
                    
                    params_list = []
                    if method_params:
                        params_list = [p.strip().split('=')[0].strip() for p in method_params.split(',')]
                    
                    # Skip private methods
                    if method_name.startswith('_'):
                        continue
                    
                    methods.append({
                        'name': method_name,
                        'params': params_list
                    })
            
            documented_classes.append({
                'name': name,
                'extends': extends,
                'description': description,
                'methods': methods
            })
        
        return documented_classes
    
    def _generate_markdown_docs(self, docs: Dict[str, Any], language: str) -> str:
        """Generate markdown documentation from the documentation dictionary."""
        markdown = []
        
        # Add title
        markdown.append("# Code Documentation\n")
        
        # Add overview
        if 'overview' in docs:
            markdown.append("## Overview\n")
            markdown.append(f"{docs['overview']}\n")
        
        # Add functions documentation
        if 'functions' in docs:
            markdown.append("## Functions\n")
            for func in docs['functions']:
                markdown.append(f"### {func['name']}\n")
                markdown.append(f"{func['description']}\n")
                
                if func['params']:
                    markdown.append("#### Parameters\n")
                    for param in func['params']:
                        markdown.append(f"- `{param}`\n")
                
                if func['returns']:
                    markdown.append("#### Returns\n")
                    markdown.append(f"{func['returns']}\n")
        
        # Add classes documentation
        if 'classes' in docs:
            markdown.append("## Classes\n")
            for cls in docs['classes']:
                markdown.append(f"### {cls['name']}\n")
                if cls.get('extends'):
                    markdown.append(f"*Extends: {cls['extends']}*\n")
                markdown.append(f"{cls['description']}\n")
                
                if cls['methods']:
                    markdown.append("#### Methods\n")
                    for method in cls['methods']:
                        markdown.append(f"##### {method['name']}\n")
                        if method['params']:
                            markdown.append("###### Parameters\n")
                            for param in method['params']:
                                markdown.append(f"- `{param}`\n")
        
        return "\n".join(markdown)

class CodeGenerator(Agent):
    """Agent that generates code based on requirements and problem analysis."""
    
    def __init__(self, name: str = "codegen"):
        super().__init__(name)
        self.supported_languages = ['python', 'javascript', 'java']
        self.common_patterns = {
            'calculator': self._generate_calculator,
            'api': self._generate_api,
            'database': self._generate_database,
            'web': self._generate_web,
            'data_processing': self._generate_data_processing
        }
        
    async def arun(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate code based on requirements and analysis."""
        requirements = context.get('requirements', '')
        if not requirements:
            return {
                'status': 'error',
                'message': 'No requirements provided for code generation'
            }
            
        # Include codebase awareness in analysis
        analysis = self._analyze_requirements(requirements)
        
        # Incorporate project info if available
        if 'project_info' in context:
            analysis['project_info'] = context['project_info']
            analysis['project_type'] = context.get('project_type', 'generic')
            analysis['existing_dependencies'] = context.get('dependencies', [])
            
        # Include file list if available
        if 'file_list' in context:
            analysis['file_list'] = context['file_list']
            
        # Check for web search results in context
        if 'web_search_results' in context:
            # Incorporate web search results into analysis
            analysis['web_search_context'] = context['web_search_results']
            
        # Extract code blocks from AI response if available
        if 'ai_response' in context and '```' in context['ai_response']:
            code_files = self._extract_code_blocks_from_response(context['ai_response'])
            if code_files:
                context['code_files'] = code_files
                
        # Generate code based on the analysis
        generated_code = self._generate_code(analysis)
        
        # Add the generated code to the context for other agents
        context['code'] = generated_code
        
        # Save generated code files if they are provided in a structured format
        if 'code_files' in context:
            # Check if files match the project type
            if 'project_type' in analysis and analysis['project_type'] != 'generic':
                self._validate_code_files_for_project(context['code_files'], analysis['project_type'])
            
        # Parse code blocks from plain text response if available
        if 'ai_response' in context and '```' in context['ai_response'] and not context.get('code_files'):
            code_files = self._extract_code_blocks_from_response(context['ai_response'])
            if code_files:
                context['code_files'] = code_files
        
        # Execute any commands if present in the requirements
        if 'commands' in context:
            command_results = await self._execute_commands(context['commands'])
            context['command_results'] = command_results
            
            # If commands were executed successfully, update the context
            if all(result['success'] for result in command_results):
                context['status'] = 'success'
                context['message'] = 'Code generated and commands executed successfully'
            else:
                context['status'] = 'partial_success'
                context['message'] = 'Code generated but some commands failed'
        
        context['generated_code'] = {
            'status': 'success',
            'analysis': analysis,
            'code': generated_code,
            'language': analysis['language'],
            'components': analysis['components']
        }
        
        return context
        
    def _validate_code_files_for_project(self, code_files: List[Dict[str, str]], project_type: str):
        """Validate that code files match the project type and suggest fixes."""
        if project_type == 'nextjs':
            # Ensure proper Next.js file structure
            for file in code_files:
                path = file.get('path', '')
                # Move components to proper location
                if 'component' in path.lower() and not path.startswith('components/'):
                    file['path'] = f"components/{os.path.basename(path)}"
                # Move pages to proper location
                elif path.endswith('.js') or path.endswith('.jsx') or path.endswith('.tsx'):
                    if not any(path.startswith(dir) for dir in ['pages/', 'components/', 'lib/', 'styles/', 'public/']):
                        file['path'] = f"pages/{os.path.basename(path)}"
        
        elif project_type == 'react':
            # Ensure proper React file structure
            for file in code_files:
                path = file.get('path', '')
                # Move components to proper location
                if 'component' in path.lower() and not path.startswith('src/components/'):
                    file['path'] = f"src/components/{os.path.basename(path)}"
                # Move other JS files to src
                elif (path.endswith('.js') or path.endswith('.jsx')) and not path.startswith('src/'):
                    file['path'] = f"src/{os.path.basename(path)}"
                    
        elif project_type in ['flask', 'fastapi']:
            # Ensure proper API structure
            for file in code_files:
                path = file.get('path', '')
                # Move route files to proper location
                if 'route' in path.lower() and not path.startswith('routes/'):
                    file['path'] = f"routes/{os.path.basename(path)}"
                    
    def _analyze_requirements(self, requirements: str) -> Dict[str, Any]:
        """Analyze requirements to determine the best approach."""
        analysis = {
            'language': 'python',  # Default language
            'components': [],
            'patterns': [],
            'complexity': 'simple'
        }
        
        # Determine language based on requirements
        if any(keyword in requirements.lower() for keyword in ['web', 'frontend', 'react', 'next', 'angular']):
            analysis['language'] = 'javascript'
        elif any(keyword in requirements.lower() for keyword in ['spring', 'enterprise', 'android']):
            analysis['language'] = 'java'
            
        # Identify components needed
        if any(keyword in requirements.lower() for keyword in ['api', 'endpoint', 'rest']):
            analysis['components'].append('api')
        if any(keyword in requirements.lower() for keyword in ['database', 'store', 'persist']):
            analysis['components'].append('database')
        if any(keyword in requirements.lower() for keyword in ['web', 'ui', 'interface']):
            analysis['components'].append('web')
        if any(keyword in requirements.lower() for keyword in ['process', 'analyze', 'transform']):
            analysis['components'].append('data_processing')
            
        # Determine specific frameworks
        if 'next' in requirements.lower():
            analysis['framework'] = 'nextjs'
        elif 'react' in requirements.lower():
            analysis['framework'] = 'react'
        elif 'vue' in requirements.lower():
            analysis['framework'] = 'vue'
        elif 'angular' in requirements.lower():
            analysis['framework'] = 'angular'
        elif 'django' in requirements.lower():
            analysis['framework'] = 'django'
        elif 'flask' in requirements.lower():
            analysis['framework'] = 'flask'
        elif 'fastapi' in requirements.lower():
            analysis['framework'] = 'fastapi'
        elif 'express' in requirements.lower():
            analysis['framework'] = 'express'
            
        # Determine complexity
        if len(analysis['components']) > 2:
            analysis['complexity'] = 'complex'
        elif len(analysis['components']) > 1:
            analysis['complexity'] = 'medium'
            
        return analysis
        
    def _generate_code(self, analysis: Dict[str, Any]) -> str:
        """Generate code based on the analysis."""
        code_components = []
        
        # Generate imports based on components
        code_components.append(self._generate_imports(analysis))
        
        # Generate main code based on patterns
        for component in analysis['components']:
            if component in self.common_patterns:
                code_components.append(self.common_patterns[component](analysis))
                
        return '\n\n'.join(code_components)
        
    def _generate_imports(self, analysis: Dict[str, Any]) -> str:
        """Generate necessary imports based on components."""
        if analysis['language'] == 'python':
            imports = []
            if 'api' in analysis['components']:
                imports.append('from fastapi import FastAPI')
            if 'database' in analysis['components']:
                imports.append('from sqlalchemy import create_engine, Column, Integer, String')
                imports.append('from sqlalchemy.ext.declarative import declarative_base')
            if 'web' in analysis['components']:
                imports.append('from flask import Flask, render_template')
            if 'data_processing' in analysis['components']:
                imports.append('import pandas as pd')
                imports.append('import numpy as np')
            return '\n'.join(imports)
        return ''
        
    def _generate_calculator(self, analysis: Dict[str, Any]) -> str:
        """Generate calculator code."""
        if analysis['language'] == 'python':
            return '''
class Calculator:
    """A simple calculator class."""
    
    def add(self, a, b):
        """Add two numbers."""
        return a + b
        
    def subtract(self, a, b):
        """Subtract b from a."""
        return a - b
        
    def multiply(self, a, b):
        """Multiply two numbers."""
        return a * b
        
    def divide(self, a, b):
        """Divide a by b."""
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
'''
        return ''
        
    def _generate_api(self, analysis: Dict[str, Any]) -> str:
        """Generate API code."""
        if analysis['language'] == 'python':
            return '''
app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
async def read_item(item_id: int):
    return {"item_id": item_id}
'''
        return ''
        
    def _generate_database(self, analysis: Dict[str, Any]) -> str:
        """Generate database code."""
        if analysis['language'] == 'python':
            return '''
Base = declarative_base()

class User(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True)
    name = Column(String)
    email = Column(String)
    
engine = create_engine('sqlite:///database.db')
Base.metadata.create_all(engine)
'''
        return ''
        
    def _generate_web(self, analysis: Dict[str, Any]) -> str:
        """Generate web interface code."""
        if analysis['language'] == 'python':
            return '''
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
'''
        return ''
        
    def _generate_data_processing(self, analysis: Dict[str, Any]) -> str:
        """Generate data processing code."""
        if analysis['language'] == 'python':
            return '''
def process_data(data):
    """Process and analyze data."""
    df = pd.DataFrame(data)
    
    # Basic statistics
    stats = {
        'mean': df.mean(),
        'median': df.median(),
        'std': df.std()
    }
    
    return stats
'''
        return ''

    async def _save_code_files(self, code_files: List[Dict[str, str]]):
        """Save a list of code files to the filesystem."""
        for file_info in code_files:
            try:
                # Get file information
                file_path = file_info.get('path')
                content = file_info.get('content')
                
                if not file_path or not content:
                    continue
                    
                # Create directory structure if needed
                dir_path = os.path.dirname(file_path)
                if dir_path and not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                    
                # Write file
                with open(file_path, 'w') as f:
                    f.write(content)
                    
                logging.info(f"Saved file: {file_path}")
                
            except Exception as e:
                logging.error(f"Error saving file {file_info.get('path')}: {str(e)}")
                
    def _extract_code_blocks_from_response(self, response: str) -> List[Dict[str, str]]:
        """Extract code blocks from AI response."""
        code_files = []
        lines = response.split('\n')
        
        current_block = None
        current_lang = None
        current_path = None
        current_content = []
        
        for line in lines:
            # Check for filename indicators
            if not current_block and '`' in line:
                # Try to extract potential filenames
                filename_match = re.search(r'[`\'"]([^`\'"]+\.\w+)[`\'"]', line)
                if filename_match:
                    current_path = filename_match.group(1)
            
            # Start of code block
            if line.startswith('```') and not current_block:
                # Extract language and possibly filename
                header = line[3:].strip()
                if ':' in header:
                    # Format might be ```language:filename
                    parts = header.split(':', 1)
                    current_lang = parts[0].strip()
                    if not current_path and len(parts) > 1:
                        current_path = parts[1].strip()
                else:
                    current_lang = header
                    
                current_block = True
                current_content = []
                continue
                
            # End of code block
            elif line.startswith('```') and current_block:
                # Determine file extension if not provided
                if current_path is None:
                    ext = self._get_extension_for_language(current_lang)
                    current_path = f"file_{len(code_files)}_{current_lang}.{ext}"
                
                # Add code file to list
                code_files.append({
                    'path': current_path,
                    'language': current_lang,
                    'content': '\n'.join(current_content)
                })
                
                # Reset tracking variables
                current_block = None
                current_lang = None
                current_path = None
                current_content = []
                continue
                
            # Inside code block
            if current_block:
                current_content.append(line)
                
        return code_files
        
    def _get_extension_for_language(self, language: str) -> str:
        """Get the file extension for a given language."""
        language = language.lower()
        extensions = {
            'python': 'py',
            'javascript': 'js',
            'typescript': 'ts',
            'html': 'html',
            'css': 'css',
            'json': 'json',
            'bash': 'sh',
            'jsx': 'jsx',
            'tsx': 'tsx',
            'markdown': 'md',
            'md': 'md',
            'yaml': 'yaml',
            'yml': 'yml',
            'java': 'java',
            'c': 'c',
            'cpp': 'cpp',
            'csharp': 'cs',
            'go': 'go',
            'ruby': 'rb',
            'php': 'php',
            'swift': 'swift',
            'kotlin': 'kt',
            'rust': 'rs',
            'scala': 'scala',
            'sql': 'sql',
            'shell': 'sh',
            'dockerfile': 'Dockerfile',
            'docker': 'Dockerfile'
        }
        
        return extensions.get(language, 'txt')

    async def _execute_commands(self, commands: List[str]) -> List[Dict[str, Any]]:
        """Execute a list of commands and return their results."""
        results = []
        for command in commands:
            try:
                # Create a subprocess
                process = await asyncio.create_subprocess_exec(
                    *shlex.split(command),
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                # Wait for the process to complete
                stdout, stderr = await process.communicate()
                
                # Add result to list
                results.append({
                    'command': command,
                    'success': process.returncode == 0,
                    'stdout': stdout.decode() if stdout else '',
                    'stderr': stderr.decode() if stderr else '',
                    'return_code': process.returncode
                })
                
            except Exception as e:
                results.append({
                    'command': command,
                    'success': False,
                    'error': str(e)
                })
                
        return results

# Update the main function to include the code generator
async def main():
    # Create a flow with all agents
    flow = AgentFlow()
    
    # Initialize agents
    analyzer = CodeAnalyzer()
    reviewer = CodeReviewer()
    optimizer = CodeOptimizer()
    docgen = DocGenerator()
    codegen = CodeGenerator()
    
    # Add agents to flow
    flow.add_agent(codegen)  # Add codegen first
    flow.add_agent(analyzer)
    flow.add_agent(reviewer)
    flow.add_agent(optimizer)
    flow.add_agent(docgen)
    
    # Set up dependencies
    flow.add_dependency("analyzer", "codegen")  # Analyzer depends on codegen
    flow.add_dependency("reviewer", "analyzer")
    flow.add_dependency("optimizer", "analyzer")
    flow.add_dependency("docgen", "analyzer")
    
    # Example requirements for code generation
    requirements = "Create a REST API for a user management system with database storage"
    
    # Create context
    context = {
        'requirements': requirements
    }
    
    # Run the flow
    result = await flow.arun(context)
    
    # Print results
    print(json.dumps(result, indent=2))
    
    # Save generated code to a file
    if 'generated_code' in result and result['generated_code']['status'] == 'success':
        with open('generated_code.py', 'w') as f:
            f.write(result['generated_code']['code'])
        print("\nGenerated code saved to 'generated_code.py'")

if __name__ == "__main__":
    asyncio.run(main())