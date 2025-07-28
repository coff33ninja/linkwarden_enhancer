#!/usr/bin/env python3
"""
Architecture Detection Script

This script analyzes the current codebase to understand:
- Module structure and organization
- Import/export relationships
- Function and class definitions
- Dependencies between modules
- Current architecture patterns

This helps create specs that build upon existing code rather than replacing it.
"""

import ast
import os
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import argparse
import html


@dataclass
class ModuleInfo:
    """Information about a Python module"""
    path: str
    name: str
    imports: List[str] = field(default_factory=list)
    from_imports: Dict[str, List[str]] = field(default_factory=dict)
    exports: List[str] = field(default_factory=list)  # __all__ or public functions/classes
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    constants: List[str] = field(default_factory=list)
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    docstring: Optional[str] = None
    lines_of_code: int = 0


@dataclass
class ArchitectureAnalysis:
    """Complete architecture analysis results"""
    modules: Dict[str, ModuleInfo] = field(default_factory=dict)
    dependency_graph: Dict[str, Set[str]] = field(default_factory=dict)
    circular_dependencies: List[List[str]] = field(default_factory=list)
    entry_points: List[str] = field(default_factory=list)
    leaf_modules: List[str] = field(default_factory=list)
    package_structure: Dict[str, List[str]] = field(default_factory=dict)
    external_dependencies: Set[str] = field(default_factory=set)
    total_lines: int = 0
    total_modules: int = 0


class ArchitectureDetector:
    """Detects and analyzes Python codebase architecture"""
    
    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path).resolve()
        self.analysis = ArchitectureAnalysis()
        self.python_files = []
        self.builtin_modules = set(sys.builtin_module_names)
        
    def analyze(self) -> ArchitectureAnalysis:
        """Perform complete architecture analysis"""
        print("Starting architecture detection...")
        
        # Step 1: Discover all Python files
        self._discover_python_files()
        print(f"Found {len(self.python_files)} Python files")
        
        # Step 2: Analyze each module
        self._analyze_modules()
        print(f"Analyzed {len(self.analysis.modules)} modules")
        
        # Step 3: Build dependency graph
        self._build_dependency_graph()
        print("Built dependency graph")
        
        # Step 4: Detect patterns and issues
        self._detect_patterns()
        print("Detected architecture patterns")
        
        # Step 5: Calculate statistics
        self._calculate_statistics()
        print("Calculated statistics")
        
        return self.analysis
    
    def _discover_python_files(self):
        """Discover all Python files in the codebase"""
        for root, dirs, files in os.walk(self.root_path):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = Path(root) / file
                    self.python_files.append(file_path)
    
    def _analyze_modules(self):
        """Analyze each Python module"""
        for file_path in self.python_files:
            try:
                module_info = self._analyze_single_module(file_path)
                if module_info:
                    self.analysis.modules[module_info.name] = module_info
            except Exception as e:
                print(f"Warning: Error analyzing {file_path}: {e}")
    
    def _analyze_single_module(self, file_path: Path) -> Optional[ModuleInfo]:
        """Analyze a single Python module"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Parse AST
            tree = ast.parse(content)
            
            # Get module name relative to root
            relative_path = file_path.relative_to(self.root_path)
            module_name = str(relative_path).replace('/', '.').replace('\\', '.').replace('.py', '')
            
            module_info = ModuleInfo(
                path=str(relative_path),
                name=module_name,
                lines_of_code=len(content.splitlines())
            )
            
            # Extract module docstring
            if (isinstance(tree.body[0], ast.Expr) and 
                isinstance(tree.body[0].value, ast.Constant) and 
                isinstance(tree.body[0].value.value, str)):
                module_info.docstring = tree.body[0].value.value
            
            # Analyze AST nodes
            for node in ast.walk(tree):
                self._analyze_ast_node(node, module_info)
            
            return module_info
            
        except Exception as e:
            print(f"Warning: Error parsing {file_path}: {e}")
            return None
    
    def _analyze_ast_node(self, node: ast.AST, module_info: ModuleInfo):
        """Analyze a single AST node"""
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_info.imports.append(alias.name)
                if not self._is_builtin_or_standard(alias.name):
                    module_info.dependencies.add(alias.name)
                    
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                module_info.from_imports[node.module] = [alias.name for alias in node.names]
                if not self._is_builtin_or_standard(node.module):
                    module_info.dependencies.add(node.module)
                    
        elif isinstance(node, ast.FunctionDef):
            if not node.name.startswith('_'):  # Public function
                module_info.functions.append(node.name)
                module_info.exports.append(node.name)
                
        elif isinstance(node, ast.AsyncFunctionDef):
            if not node.name.startswith('_'):  # Public async function
                module_info.functions.append(f"async {node.name}")
                module_info.exports.append(node.name)
                
        elif isinstance(node, ast.ClassDef):
            if not node.name.startswith('_'):  # Public class
                module_info.classes.append(node.name)
                module_info.exports.append(node.name)
                
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    if target.id.isupper():  # Constant
                        module_info.constants.append(target.id)
                    elif target.id == '__all__':
                        # Extract __all__ exports
                        if isinstance(node.value, ast.List):
                            module_info.exports = [
                                elt.s if isinstance(elt, ast.Str) else elt.value
                                for elt in node.value.elts
                                if isinstance(elt, (ast.Str, ast.Constant))
                            ]
    
    def _is_builtin_or_standard(self, module_name: str) -> bool:
        """Check if module is builtin or standard library"""
        if not module_name:
            return True
            
        # Split module name to get top-level package
        top_level = module_name.split('.')[0]
        
        # Check if it's a builtin module
        if top_level in self.builtin_modules:
            return True
            
        # Check if it's a standard library module (common ones)
        standard_lib = {
            'os', 'sys', 'json', 'ast', 'pathlib', 'typing', 'dataclasses',
            'collections', 'itertools', 'functools', 'operator', 'datetime',
            'time', 're', 'math', 'random', 'urllib', 'http', 'email',
            'html', 'xml', 'csv', 'sqlite3', 'logging', 'argparse',
            'configparser', 'subprocess', 'threading', 'multiprocessing',
            'asyncio', 'concurrent', 'queue', 'socket', 'ssl', 'hashlib',
            'base64', 'uuid', 'pickle', 'copy', 'weakref', 'gc', 'inspect'
        }
        
        return top_level in standard_lib
    
    def _build_dependency_graph(self):
        """Build module dependency graph"""
        for module_name, module_info in self.analysis.modules.items():
            self.analysis.dependency_graph[module_name] = set()
            
            # Add internal dependencies
            for dep in module_info.dependencies:
                # Check if dependency is an internal module
                internal_dep = self._find_internal_module(dep)
                if internal_dep:
                    self.analysis.dependency_graph[module_name].add(internal_dep)
                    # Add reverse dependency
                    if internal_dep in self.analysis.modules:
                        self.analysis.modules[internal_dep].dependents.add(module_name)
                else:
                    # External dependency
                    self.analysis.external_dependencies.add(dep)
    
    def _find_internal_module(self, dep_name: str) -> Optional[str]:
        """Find if dependency is an internal module"""
        # Direct match
        if dep_name in self.analysis.modules:
            return dep_name
            
        # Check for partial matches (e.g., 'linkwarden_enhancer.core' -> 'linkwarden_enhancer.core.something')
        for module_name in self.analysis.modules:
            if module_name.startswith(dep_name + '.') or dep_name.startswith(module_name + '.'):
                return module_name
                
        return None
    
    def _detect_patterns(self):
        """Detect architecture patterns and issues"""
        # Find entry points (modules with main or CLI)
        for module_name, module_info in self.analysis.modules.items():
            if ('main' in module_info.functions or 
                '__main__' in module_info.name or
                'cli' in module_info.name.lower() or
                'main.py' in module_info.path):
                self.analysis.entry_points.append(module_name)
        
        # Find leaf modules (no dependents)
        for module_name, module_info in self.analysis.modules.items():
            if not module_info.dependents:
                self.analysis.leaf_modules.append(module_name)
        
        # Build package structure
        for module_name in self.analysis.modules:
            parts = module_name.split('.')
            for i in range(len(parts)):
                package = '.'.join(parts[:i+1])
                if package not in self.analysis.package_structure:
                    self.analysis.package_structure[package] = []
                if i == len(parts) - 1:  # Leaf module
                    parent_package = '.'.join(parts[:-1]) if len(parts) > 1 else 'root'
                    if parent_package not in self.analysis.package_structure:
                        self.analysis.package_structure[parent_package] = []
                    self.analysis.package_structure[parent_package].append(module_name)
        
        # Detect circular dependencies (simplified)
        self._detect_circular_dependencies()
    
    def _detect_circular_dependencies(self):
        """Detect circular dependencies using DFS"""
        visited = set()
        rec_stack = set()
        
        def dfs(node, path):
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                self.analysis.circular_dependencies.append(cycle)
                return
                
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self.analysis.dependency_graph.get(node, []):
                dfs(neighbor, path + [node])
                
            rec_stack.remove(node)
        
        for module in self.analysis.modules:
            if module not in visited:
                dfs(module, [])
    
    def _calculate_statistics(self):
        """Calculate architecture statistics"""
        self.analysis.total_modules = len(self.analysis.modules)
        self.analysis.total_lines = sum(
            module.lines_of_code for module in self.analysis.modules.values()
        )
    
    def generate_report(self, output_format: str = 'text') -> str:
        """Generate architecture analysis report"""
        if output_format == 'json':
            return self._generate_json_report()
        elif output_format == 'html':
            return self._generate_html_report()
        else:
            return self._generate_text_report()
    
    def _generate_text_report(self) -> str:
        """Generate human-readable text report"""
        report = []
        report.append("ARCHITECTURE ANALYSIS REPORT")
        report.append("=" * 50)
        
        # Overview
        report.append(f"\nOVERVIEW")
        report.append(f"Total modules: {self.analysis.total_modules}")
        report.append(f"Total lines of code: {self.analysis.total_lines:,}")
        report.append(f"External dependencies: {len(self.analysis.external_dependencies)}")
        report.append(f"Entry points: {len(self.analysis.entry_points)}")
        report.append(f"Circular dependencies: {len(self.analysis.circular_dependencies)}")
        
        # Package structure
        report.append(f"\nPACKAGE STRUCTURE")
        for package, modules in sorted(self.analysis.package_structure.items()):
            if package != 'root' and modules:
                report.append(f"  {package}/")
                for module in sorted(modules):
                    if module != package:  # Don't show package as its own child
                        report.append(f"    └── {module.split('.')[-1]}")
        
        # Entry points
        if self.analysis.entry_points:
            report.append(f"\nENTRY POINTS")
            for entry_point in self.analysis.entry_points:
                module = self.analysis.modules[entry_point]
                report.append(f"  • {entry_point} ({module.path})")
                if module.functions:
                    report.append(f"    Functions: {', '.join(module.functions[:5])}")
        
        # Top modules by size
        report.append(f"\nLARGEST MODULES")
        largest_modules = sorted(
            self.analysis.modules.items(),
            key=lambda x: x[1].lines_of_code,
            reverse=True
        )[:10]
        
        for module_name, module_info in largest_modules:
            report.append(f"  • {module_name}: {module_info.lines_of_code} lines")
            if module_info.classes:
                report.append(f"    Classes: {', '.join(module_info.classes[:3])}")
            if module_info.functions:
                report.append(f"    Functions: {', '.join(module_info.functions[:3])}")
        
        # Most connected modules
        report.append(f"\nMOST CONNECTED MODULES")
        most_connected = sorted(
            self.analysis.modules.items(),
            key=lambda x: len(x[1].dependencies) + len(x[1].dependents),
            reverse=True
        )[:10]
        
        for module_name, module_info in most_connected:
            total_connections = len(module_info.dependencies) + len(module_info.dependents)
            report.append(f"  • {module_name}: {total_connections} connections")
            report.append(f"    Dependencies: {len(module_info.dependencies)}, Dependents: {len(module_info.dependents)}")
        
        # External dependencies
        if self.analysis.external_dependencies:
            report.append(f"\nEXTERNAL DEPENDENCIES")
            for dep in sorted(self.analysis.external_dependencies):
                report.append(f"  • {dep}")
        
        # Circular dependencies
        if self.analysis.circular_dependencies:
            report.append(f"\nCIRCULAR DEPENDENCIES")
            for i, cycle in enumerate(self.analysis.circular_dependencies):
                report.append(f"  {i+1}. {' → '.join(cycle)}")
        
        # Module details
        report.append(f"\nMODULE DETAILS")
        for module_name, module_info in sorted(self.analysis.modules.items()):
            report.append(f"\n  {module_name}")
            report.append(f"     Path: {module_info.path}")
            report.append(f"     Lines: {module_info.lines_of_code}")
            
            if module_info.docstring:
                doc_preview = module_info.docstring.split('\n')[0][:80]
                report.append(f"     Doc: {doc_preview}...")
            
            if module_info.classes:
                report.append(f"     Classes: {', '.join(module_info.classes)}")
            
            if module_info.functions:
                report.append(f"     Functions: {', '.join(module_info.functions)}")
            
            if module_info.imports:
                report.append(f"     Imports: {', '.join(module_info.imports[:5])}")
                if len(module_info.imports) > 5:
                    report.append(f"              ... and {len(module_info.imports) - 5} more")
            
            if module_info.dependencies:
                internal_deps = [dep for dep in module_info.dependencies 
                               if self._find_internal_module(dep)]
                if internal_deps:
                    report.append(f"     Internal deps: {', '.join(internal_deps)}")
        
        return '\n'.join(report)
    
    def _generate_json_report(self) -> str:
        """Generate JSON report for programmatic use"""
        # Convert dataclasses to dictionaries
        def convert_to_dict(obj):
            if hasattr(obj, '__dict__'):
                result = {}
                for key, value in obj.__dict__.items():
                    if isinstance(value, set):
                        result[key] = list(value)
                    elif isinstance(value, dict):
                        result[key] = {k: convert_to_dict(v) for k, v in value.items()}
                    elif isinstance(value, list):
                        result[key] = [convert_to_dict(item) for item in value]
                    else:
                        result[key] = convert_to_dict(value)
                return result
            elif isinstance(obj, set):
                return list(obj)
            elif isinstance(obj, list):
                return [convert_to_dict(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_dict(v) for k, v in obj.items()}
            else:
                return obj
        
        return json.dumps(convert_to_dict(self.analysis), indent=2)
    
    def _generate_html_report(self) -> str:
        """Generate interactive HTML report with visual diagrams"""
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Architecture Analysis Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        
        .header {{
            background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        
        .stats {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            padding: 30px;
            background: #f8f9fa;
        }}
        
        .stat-card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 5px 15px rgba(0,0,0,0.08);
            transition: transform 0.3s ease;
        }}
        
        .stat-card:hover {{
            transform: translateY(-5px);
        }}
        
        .stat-number {{
            font-size: 2.5em;
            font-weight: bold;
            color: #3498db;
            margin: 0;
        }}
        
        .stat-label {{
            color: #7f8c8d;
            margin: 5px 0 0 0;
            font-size: 0.9em;
        }}
        
        .content {{
            padding: 30px;
        }}
        
        .section {{
            margin-bottom: 40px;
        }}
        
        .section h2 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        
        .architecture-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }}
        
        .module-tree {{
            background: #f8f9fa;
            border-radius: 10px;
            padding: 20px;
            font-family: 'Courier New', monospace;
            overflow-x: auto;
        }}
        
        .venv-item {{
            color: #e74c3c;
            font-weight: bold;
        }}
        
        .core-module {{
            color: #2980b9;
            font-weight: bold;
        }}
        
        .utility-module {{
            color: #27ae60;
        }}
        
        .cli-module {{
            color: #8e44ad;
            font-weight: bold;
        }}
        
        .ai-module {{
            color: #f39c12;
            font-weight: bold;
        }}
        
        .dependency-graph {{
            background: white;
            border: 2px solid #ecf0f1;
            border-radius: 10px;
            padding: 20px;
            min-height: 400px;
        }}
        
        .module-card {{
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .module-card:hover {{
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }}
        
        .module-name {{
            font-weight: bold;
            color: #2c3e50;
            font-size: 1.1em;
            margin-bottom: 8px;
        }}
        
        .module-path {{
            color: #7f8c8d;
            font-size: 0.9em;
            font-family: monospace;
            margin-bottom: 10px;
        }}
        
        .module-stats {{
            display: flex;
            gap: 15px;
            margin-bottom: 10px;
        }}
        
        .module-stat {{
            background: #ecf0f1;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
        }}
        
        .dependencies {{
            margin-top: 10px;
        }}
        
        .dep-list {{
            display: flex;
            flex-wrap: wrap;
            gap: 5px;
            margin-top: 5px;
        }}
        
        .dep-tag {{
            background: #3498db;
            color: white;
            padding: 2px 6px;
            border-radius: 3px;
            font-size: 0.7em;
        }}
        
        .external-dep {{
            background: #e74c3c;
        }}
        
        .internal-dep {{
            background: #27ae60;
        }}
        
        .entry-point {{
            border-left: 4px solid #f39c12;
            background: #fef9e7;
        }}
        
        .circular-warning {{
            background: #fdf2f2;
            border: 1px solid #f56565;
            border-radius: 8px;
            padding: 15px;
            margin: 10px 0;
        }}
        
        .warning-title {{
            color: #e53e3e;
            font-weight: bold;
            margin-bottom: 10px;
        }}
        
        .cycle-path {{
            font-family: monospace;
            background: #fed7d7;
            padding: 5px;
            border-radius: 4px;
            margin: 5px 0;
        }}
        
        .toggle-btn {{
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 15px;
            border-radius: 5px;
            cursor: pointer;
            margin: 5px;
            transition: background 0.3s ease;
        }}
        
        .toggle-btn:hover {{
            background: #2980b9;
        }}
        
        .hidden {{
            display: none;
        }}
        
        .legend {{
            background: #f8f9fa;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
        }}
        
        .legend-item {{
            display: inline-block;
            margin: 5px 10px;
            font-size: 0.9em;
        }}
        
        .legend-color {{
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 2px;
            margin-right: 5px;
            vertical-align: middle;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏗️ Architecture Analysis Report</h1>
            <p>Generated on {self._get_timestamp()}</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{self.analysis.total_modules}</div>
                <div class="stat-label">Total Modules</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{self.analysis.total_lines:,}</div>
                <div class="stat-label">Lines of Code</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(self.analysis.external_dependencies)}</div>
                <div class="stat-label">External Dependencies</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(self.analysis.entry_points)}</div>
                <div class="stat-label">Entry Points</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{len(self.analysis.circular_dependencies)}</div>
                <div class="stat-label">Circular Dependencies</div>
            </div>
        </div>
        
        <div class="content">
            {self._generate_html_architecture_overview()}
            {self._generate_html_modules_section()}
            {self._generate_html_dependencies_section()}
            {self._generate_html_issues_section()}
        </div>
    </div>
    
    <script>
        function toggleSection(sectionId) {{
            const section = document.getElementById(sectionId);
            section.classList.toggle('hidden');
        }}
        
        function filterModules(type) {{
            const modules = document.querySelectorAll('.module-card');
            modules.forEach(module => {{
                if (type === 'all') {{
                    module.style.display = 'block';
                }} else {{
                    const hasClass = module.classList.contains(type + '-module');
                    module.style.display = hasClass ? 'block' : 'none';
                }}
            }});
        }}
    </script>
</body>
</html>
        """
        return html_content
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for report"""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _generate_html_architecture_overview(self) -> str:
        """Generate HTML architecture overview section"""
        content = '<div class="section">'
        content += '<h2>📁 Project Structure</h2>'
        
        # Legend
        content += '''
        <div class="legend">
            <strong>Module Types:</strong>
            <span class="legend-item">
                <span class="legend-color venv-item"></span>Virtual Environment
            </span>
            <span class="legend-item">
                <span class="legend-color core-module"></span>Core Modules
            </span>
            <span class="legend-item">
                <span class="legend-color ai-module"></span>AI Modules
            </span>
            <span class="legend-item">
                <span class="legend-color cli-module"></span>CLI Modules
            </span>
            <span class="legend-item">
                <span class="legend-color utility-module"></span>Utility Modules
            </span>
        </div>
        '''
        
        content += '<div class="architecture-grid">'
        
        # Module tree
        content += '<div class="module-tree">'
        content += '<h3>📂 Directory Structure</h3>'
        content += self._generate_directory_tree()
        content += '</div>'
        
        # Dependency overview
        content += '<div class="dependency-graph">'
        content += '<h3>🔗 Dependency Overview</h3>'
        content += self._generate_dependency_overview()
        content += '</div>'
        
        content += '</div>'
        content += '</div>'
        
        return content
    
    def _generate_directory_tree(self) -> str:
        """Generate visual directory tree"""
        tree_lines = []
        
        # Group modules by directory
        dir_structure = defaultdict(list)
        
        for module_name, module_info in self.analysis.modules.items():
            path_parts = Path(module_info.path).parts
            if len(path_parts) > 1:
                directory = path_parts[0]
                dir_structure[directory].append((module_name, module_info))
            else:
                dir_structure['root'].append((module_name, module_info))
        
        # Special handling for venv
        if any('venv' in d or '.venv' in d for d in dir_structure.keys()):
            tree_lines.append('<span class="venv-item">📁 .venv/ (Virtual Environment)</span>')
            tree_lines.append('<span class="venv-item">  └── [Python packages and dependencies]</span>')
            tree_lines.append('')
        
        # Generate tree for each directory
        for directory in sorted(dir_structure.keys()):
            if directory in ['venv', '.venv']:
                continue  # Already handled above
                
            modules = dir_structure[directory]
            if directory == 'root':
                tree_lines.append('📁 Root Files:')
            else:
                tree_lines.append(f'📁 {directory}/')
            
            for i, (module_name, module_info) in enumerate(sorted(modules)):
                is_last = i == len(modules) - 1
                prefix = '  └── ' if is_last else '  ├── '
                
                # Determine module type and color
                module_class = self._get_module_class(module_name, module_info)
                file_name = Path(module_info.path).name
                
                tree_lines.append(f'{prefix}<span class="{module_class}">{file_name}</span>')
                
                # Add brief info
                if module_info.classes or module_info.functions:
                    info_parts = []
                    if module_info.classes:
                        info_parts.append(f"{len(module_info.classes)} classes")
                    if module_info.functions:
                        info_parts.append(f"{len(module_info.functions)} functions")
                    
                    indent = '      ' if is_last else '  │   '
                    tree_lines.append(f'{indent}({", ".join(info_parts)})')
        
        return '<br>'.join(tree_lines)
    
    def _get_module_class(self, module_name: str, module_info) -> str:
        """Determine CSS class for module based on its type"""
        path_lower = module_info.path.lower()
        name_lower = module_name.lower()
        
        if 'venv' in path_lower or '.venv' in path_lower:
            return 'venv-item'
        elif 'ai' in path_lower or 'ai' in name_lower:
            return 'ai-module'
        elif 'cli' in path_lower or 'cli' in name_lower or 'main' in name_lower:
            return 'cli-module'
        elif any(core in name_lower for core in ['core', 'engine', 'analyzer', 'processor']):
            return 'core-module'
        else:
            return 'utility-module'
    
    def _generate_dependency_overview(self) -> str:
        """Generate dependency overview visualization"""
        content = []
        
        # Entry points
        if self.analysis.entry_points:
            content.append('<h4>🚀 Entry Points</h4>')
            for entry_point in self.analysis.entry_points:
                module = self.analysis.modules[entry_point]
                content.append(f'<div class="dep-tag internal-dep">{Path(module.path).name}</div>')
        
        # Most connected modules
        content.append('<h4>🔗 Most Connected Modules</h4>')
        most_connected = sorted(
            self.analysis.modules.items(),
            key=lambda x: len(x[1].dependencies) + len(x[1].dependents),
            reverse=True
        )[:5]
        
        for module_name, module_info in most_connected:
            total_connections = len(module_info.dependencies) + len(module_info.dependents)
            content.append(f'<div style="margin: 5px 0;">')
            content.append(f'<strong>{Path(module_info.path).name}</strong>: {total_connections} connections')
            content.append(f'<br><small>{len(module_info.dependencies)} deps, {len(module_info.dependents)} dependents</small>')
            content.append('</div>')
        
        return '<br>'.join(content)
    
    def _generate_html_modules_section(self) -> str:
        """Generate HTML modules section"""
        content = '<div class="section">'
        content += '<h2>📋 Module Details</h2>'
        
        # Filter buttons
        content += '''
        <div style="margin-bottom: 20px;">
            <button class="toggle-btn" onclick="filterModules('all')">All Modules</button>
            <button class="toggle-btn" onclick="filterModules('core')">Core</button>
            <button class="toggle-btn" onclick="filterModules('ai')">AI</button>
            <button class="toggle-btn" onclick="filterModules('cli')">CLI</button>
            <button class="toggle-btn" onclick="filterModules('utility')">Utilities</button>
        </div>
        '''
        
        # Module cards
        for module_name, module_info in sorted(self.analysis.modules.items()):
            if 'venv' in module_info.path.lower() or '.venv' in module_info.path.lower():
                continue  # Skip venv modules
                
            module_class = self._get_module_class(module_name, module_info)
            entry_point_class = ' entry-point' if module_name in self.analysis.entry_points else ''
            
            content += f'<div class="module-card {module_class}{entry_point_class}">'
            
            # Module header
            content += f'<div class="module-name">{html.escape(module_name)}</div>'
            content += f'<div class="module-path">{html.escape(module_info.path)}</div>'
            
            # Stats
            content += '<div class="module-stats">'
            content += f'<span class="module-stat">{module_info.lines_of_code} lines</span>'
            if module_info.classes:
                content += f'<span class="module-stat">{len(module_info.classes)} classes</span>'
            if module_info.functions:
                content += f'<span class="module-stat">{len(module_info.functions)} functions</span>'
            content += '</div>'
            
            # Docstring
            if module_info.docstring:
                doc_preview = module_info.docstring.split('\\n')[0][:100]
                content += f'<div style="font-style: italic; color: #666; margin: 10px 0;">{html.escape(doc_preview)}...</div>'
            
            # Classes and functions
            if module_info.classes:
                content += f'<div><strong>Classes:</strong> {", ".join(module_info.classes[:5])}'
                if len(module_info.classes) > 5:
                    content += f' <small>... and {len(module_info.classes) - 5} more</small>'
                content += '</div>'
            
            if module_info.functions:
                content += f'<div><strong>Functions:</strong> {", ".join(module_info.functions[:5])}'
                if len(module_info.functions) > 5:
                    content += f' <small>... and {len(module_info.functions) - 5} more</small>'
                content += '</div>'
            
            # Dependencies
            if module_info.dependencies or module_info.dependents:
                content += '<div class="dependencies">'
                
                if module_info.dependencies:
                    internal_deps = [dep for dep in module_info.dependencies 
                                   if self._find_internal_module(dep)]
                    external_deps = [dep for dep in module_info.dependencies 
                                   if not self._find_internal_module(dep)]
                    
                    if internal_deps:
                        content += '<div><strong>Internal Dependencies:</strong></div>'
                        content += '<div class="dep-list">'
                        for dep in internal_deps[:10]:
                            content += f'<span class="dep-tag internal-dep">{html.escape(dep)}</span>'
                        content += '</div>'
                    
                    if external_deps:
                        content += '<div><strong>External Dependencies:</strong></div>'
                        content += '<div class="dep-list">'
                        for dep in external_deps[:10]:
                            content += f'<span class="dep-tag external-dep">{html.escape(dep)}</span>'
                        content += '</div>'
                
                if module_info.dependents:
                    content += '<div><strong>Used by:</strong></div>'
                    content += '<div class="dep-list">'
                    for dep in list(module_info.dependents)[:10]:
                        content += f'<span class="dep-tag internal-dep">{html.escape(dep)}</span>'
                    content += '</div>'
                
                content += '</div>'
            
            content += '</div>'
        
        content += '</div>'
        return content
    
    def _generate_html_dependencies_section(self) -> str:
        """Generate HTML dependencies section"""
        content = '<div class="section">'
        content += '<h2>📦 External Dependencies</h2>'
        
        if self.analysis.external_dependencies:
            content += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 10px;">'
            for dep in sorted(self.analysis.external_dependencies):
                content += f'<div class="dep-tag external-dep" style="padding: 10px; text-align: center;">{html.escape(dep)}</div>'
            content += '</div>'
        else:
            content += '<p>No external dependencies detected.</p>'
        
        content += '</div>'
        return content
    
    def _generate_html_issues_section(self) -> str:
        """Generate HTML issues section"""
        content = '<div class="section">'
        content += '<h2>⚠️ Potential Issues</h2>'
        
        if self.analysis.circular_dependencies:
            content += '<div class="circular-warning">'
            content += '<div class="warning-title">Circular Dependencies Detected</div>'
            content += '<p>The following circular dependencies were found in your codebase:</p>'
            
            for i, cycle in enumerate(self.analysis.circular_dependencies):
                content += f'<div class="cycle-path">{i+1}. {" → ".join(cycle)}</div>'
            
            content += '</div>'
        else:
            content += '<div style="color: #27ae60; padding: 15px; background: #d5f4e6; border-radius: 8px;">'
            content += '✅ No circular dependencies detected!'
            content += '</div>'
        
        content += '</div>'
        return content
    
    def save_report(self, filename: str, format: str = 'text'):
        """Save report to file"""
        report = self.generate_report(format)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to {filename}")


def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description='Analyze Python codebase architecture')
    parser.add_argument('path', nargs='?', default='.', help='Path to analyze (default: current directory)')
    parser.add_argument('--format', choices=['text', 'json', 'html'], default='text', help='Output format')
    parser.add_argument('--output', '-o', help='Output file (default: print to stdout)')
    parser.add_argument('--save-json', help='Save JSON report to file')
    
    args = parser.parse_args()
    
    # Analyze architecture
    detector = ArchitectureDetector(args.path)
    analysis = detector.analyze()
    
    # Generate and output report
    report = detector.generate_report(args.format)
    
    if args.output:
        detector.save_report(args.output, args.format)
    else:
        print(report)
    
    # Save JSON if requested
    if args.save_json:
        detector.save_report(args.save_json, 'json')
    
    print(f"\nAnalysis complete! Found {analysis.total_modules} modules with {analysis.total_lines:,} lines of code.")


if __name__ == '__main__':
    main()