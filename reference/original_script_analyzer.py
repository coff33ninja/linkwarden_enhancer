"""
Original Script Analyzer - Read-only analysis of original bookmark management scripts.
Extracts patterns, algorithms, and insights without modifying the original code.
"""

import ast
import re
import json
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import inspect

from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class FunctionAnalysis:
    """Analysis of a function in the original script"""
    name: str
    line_number: int
    docstring: Optional[str]
    parameters: List[str]
    return_type: Optional[str]
    complexity_score: int
    calls_made: List[str]
    variables_used: Set[str]
    patterns_detected: List[str]
    purpose: str = ""
    algorithm_type: str = ""


@dataclass
class ClassAnalysis:
    """Analysis of a class in the original script"""
    name: str
    line_number: int
    docstring: Optional[str]
    methods: List[FunctionAnalysis]
    attributes: List[str]
    inheritance: List[str]
    patterns_detected: List[str]
    purpose: str = ""


@dataclass
class PatternAnalysis:
    """Analysis of a detected pattern in the original script"""
    pattern_type: str  # 'tag_normalization', 'collection_organization', 'suggestion_mechanism'
    pattern_name: str
    description: str
    code_location: str
    line_numbers: List[int]
    algorithm_description: str
    input_format: str
    output_format: str
    complexity: str  # 'simple', 'moderate', 'complex'
    reusability: str  # 'high', 'medium', 'low'
    modernization_suggestions: List[str]


@dataclass
class ScriptAnalysisResult:
    """Complete analysis result of the original script"""
    script_path: str
    analysis_timestamp: str
    total_lines: int
    total_functions: int
    total_classes: int
    
    # Code structure analysis
    functions: List[FunctionAnalysis]
    classes: List[ClassAnalysis]
    imports: List[str]
    global_variables: List[str]
    
    # Pattern analysis
    tag_normalization_patterns: List[PatternAnalysis]
    collection_organization_patterns: List[PatternAnalysis]
    suggestion_mechanism_patterns: List[PatternAnalysis]
    other_patterns: List[PatternAnalysis]
    
    # Algorithm insights
    algorithm_insights: Dict[str, Any]
    design_patterns: List[str]
    data_structures_used: List[str]
    
    # Recommendations
    modernization_recommendations: List[str]
    reusable_components: List[str]
    improvement_opportunities: List[str]
    
    # Documentation
    reference_documentation: str
    api_documentation: Dict[str, str]


class OriginalScriptAnalyzer:
    """Analyzer for extracting patterns from original bookmark management scripts"""
    
    def __init__(self):
        """Initialize the original script analyzer"""
        self.analysis_cache = {}
        
        # Pattern detection rules
        self.tag_normalization_keywords = [
            'normalize', 'clean', 'sanitize', 'format', 'standardize',
            'tag', 'label', 'category', 'strip', 'lower', 'upper'
        ]
        
        self.collection_organization_keywords = [
            'organize', 'group', 'sort', 'categorize', 'classify',
            'collection', 'folder', 'directory', 'hierarchy', 'tree'
        ]
        
        self.suggestion_mechanism_keywords = [
            'suggest', 'recommend', 'predict', 'infer', 'guess',
            'match', 'similar', 'related', 'auto', 'smart'
        ]
        
        # Algorithm pattern recognition
        self.algorithm_patterns = {
            'string_matching': ['match', 'find', 'search', 'contains', 'startswith', 'endswith'],
            'text_processing': ['split', 'join', 'replace', 'strip', 'format'],
            'data_filtering': ['filter', 'select', 'where', 'if', 'condition'],
            'sorting': ['sort', 'order', 'rank', 'priority'],
            'grouping': ['group', 'cluster', 'aggregate', 'collect'],
            'scoring': ['score', 'weight', 'rank', 'priority', 'confidence']
        }
        
        logger.info("Original script analyzer initialized")
    
    def analyze_script(self, script_path: str) -> ScriptAnalysisResult:
        """Analyze an original script and extract patterns and insights"""
        
        try:
            script_path = Path(script_path)
            
            if not script_path.exists():
                raise FileNotFoundError(f"Script not found: {script_path}")
            
            logger.info(f"Starting analysis of script: {script_path}")
            
            # Read script content
            with open(script_path, 'r', encoding='utf-8') as f:
                script_content = f.read()
            
            # Parse AST
            try:
                tree = ast.parse(script_content)
            except SyntaxError as e:
                logger.error(f"Syntax error in script: {e}")
                raise
            
            # Perform analysis
            analysis_result = self._perform_comprehensive_analysis(
                str(script_path), script_content, tree
            )
            
            logger.info(f"Analysis completed: {len(analysis_result.functions)} functions, "
                       f"{len(analysis_result.classes)} classes analyzed")
            
            return analysis_result
            
        except Exception as e:
            logger.error(f"Failed to analyze script {script_path}: {e}")
            raise
    
    def _perform_comprehensive_analysis(self, script_path: str, content: str, tree: ast.AST) -> ScriptAnalysisResult:
        """Perform comprehensive analysis of the script"""
        
        from datetime import datetime
        
        # Basic metrics
        lines = content.split('\n')
        total_lines = len(lines)
        
        # Extract components
        functions = self._extract_functions(tree, lines)
        classes = self._extract_classes(tree, lines)
        imports = self._extract_imports(tree)
        global_vars = self._extract_global_variables(tree)
        
        # Pattern analysis
        tag_patterns = self._analyze_tag_normalization_patterns(tree, content, lines)
        collection_patterns = self._analyze_collection_organization_patterns(tree, content, lines)
        suggestion_patterns = self._analyze_suggestion_mechanism_patterns(tree, content, lines)
        other_patterns = self._analyze_other_patterns(tree, content, lines)
        
        # Algorithm insights
        algorithm_insights = self._extract_algorithm_insights(functions, classes, content)
        design_patterns = self._detect_design_patterns(tree, content)
        data_structures = self._analyze_data_structures(tree, content)
        
        # Generate recommendations
        modernization_recs = self._generate_modernization_recommendations(
            functions, classes, tag_patterns, collection_patterns, suggestion_patterns
        )
        reusable_components = self._identify_reusable_components(functions, classes)
        improvements = self._identify_improvement_opportunities(
            functions, classes, algorithm_insights
        )
        
        # Generate documentation
        reference_doc = self._generate_reference_documentation(
            functions, classes, tag_patterns, collection_patterns, suggestion_patterns
        )
        api_doc = self._generate_api_documentation(functions, classes)
        
        return ScriptAnalysisResult(
            script_path=script_path,
            analysis_timestamp=datetime.now().isoformat(),
            total_lines=total_lines,
            total_functions=len(functions),
            total_classes=len(classes),
            functions=functions,
            classes=classes,
            imports=imports,
            global_variables=global_vars,
            tag_normalization_patterns=tag_patterns,
            collection_organization_patterns=collection_patterns,
            suggestion_mechanism_patterns=suggestion_patterns,
            other_patterns=other_patterns,
            algorithm_insights=algorithm_insights,
            design_patterns=design_patterns,
            data_structures_used=data_structures,
            modernization_recommendations=modernization_recs,
            reusable_components=reusable_components,
            improvement_opportunities=improvements,
            reference_documentation=reference_doc,
            api_documentation=api_doc
        )
    
    def _extract_functions(self, tree: ast.AST, lines: List[str]) -> List[FunctionAnalysis]:
        """Extract and analyze functions from the AST"""
        
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_analysis = self._analyze_function(node, lines)
                functions.append(func_analysis)
        
        return functions
    
    def _analyze_function(self, node: ast.FunctionDef, lines: List[str]) -> FunctionAnalysis:
        """Analyze a single function"""
        
        # Extract basic info
        name = node.name
        line_number = node.lineno
        docstring = ast.get_docstring(node)
        
        # Extract parameters
        parameters = [arg.arg for arg in node.args.args]
        
        # Extract return type (if annotated)
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
        
        # Calculate complexity (simplified cyclomatic complexity)
        complexity_score = self._calculate_complexity(node)
        
        # Extract function calls
        calls_made = self._extract_function_calls(node)
        
        # Extract variables used
        variables_used = self._extract_variables_used(node)
        
        # Detect patterns
        patterns_detected = self._detect_function_patterns(node, name, docstring or "")
        
        # Determine purpose and algorithm type
        purpose = self._determine_function_purpose(name, docstring, patterns_detected)
        algorithm_type = self._determine_algorithm_type(node, name, calls_made)
        
        return FunctionAnalysis(
            name=name,
            line_number=line_number,
            docstring=docstring,
            parameters=parameters,
            return_type=return_type,
            complexity_score=complexity_score,
            calls_made=calls_made,
            variables_used=variables_used,
            patterns_detected=patterns_detected,
            purpose=purpose,
            algorithm_type=algorithm_type
        )
    
    def _extract_classes(self, tree: ast.AST, lines: List[str]) -> List[ClassAnalysis]:
        """Extract and analyze classes from the AST"""
        
        classes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_analysis = self._analyze_class(node, lines)
                classes.append(class_analysis)
        
        return classes
    
    def _analyze_class(self, node: ast.ClassDef, lines: List[str]) -> ClassAnalysis:
        """Analyze a single class"""
        
        name = node.name
        line_number = node.lineno
        docstring = ast.get_docstring(node)
        
        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_analysis = self._analyze_function(item, lines)
                methods.append(method_analysis)
        
        # Extract attributes (simplified - from __init__ method)
        attributes = self._extract_class_attributes(node)
        
        # Extract inheritance
        inheritance = [ast.unparse(base) if hasattr(ast, 'unparse') else str(base) 
                      for base in node.bases]
        
        # Detect patterns
        patterns_detected = self._detect_class_patterns(node, name, docstring or "")
        
        # Determine purpose
        purpose = self._determine_class_purpose(name, docstring, methods, patterns_detected)
        
        return ClassAnalysis(
            name=name,
            line_number=line_number,
            docstring=docstring,
            methods=methods,
            attributes=attributes,
            inheritance=inheritance,
            patterns_detected=patterns_detected,
            purpose=purpose
        )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}")
        
        return imports
    
    def _extract_global_variables(self, tree: ast.AST) -> List[str]:
        """Extract global variable assignments"""
        
        global_vars = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        global_vars.append(target.id)
        
        return global_vars
    
    def _analyze_tag_normalization_patterns(self, tree: ast.AST, content: str, lines: List[str]) -> List[PatternAnalysis]:
        """Analyze tag normalization patterns in the script"""
        
        patterns = []
        
        # Look for functions that might handle tag normalization
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name.lower()
                func_content = ast.unparse(node) if hasattr(ast, 'unparse') else ""
                
                # Check if function is related to tag normalization
                if any(keyword in func_name for keyword in self.tag_normalization_keywords):
                    pattern = self._create_pattern_analysis(
                        pattern_type="tag_normalization",
                        pattern_name=f"Tag Normalization: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
                
                # Check function body for tag-related operations
                elif self._contains_tag_operations(node):
                    pattern = self._create_pattern_analysis(
                        pattern_type="tag_normalization",
                        pattern_name=f"Tag Processing: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_collection_organization_patterns(self, tree: ast.AST, content: str, lines: List[str]) -> List[PatternAnalysis]:
        """Analyze collection organization patterns in the script"""
        
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name.lower()
                func_content = ast.unparse(node) if hasattr(ast, 'unparse') else ""
                
                # Check if function is related to collection organization
                if any(keyword in func_name for keyword in self.collection_organization_keywords):
                    pattern = self._create_pattern_analysis(
                        pattern_type="collection_organization",
                        pattern_name=f"Collection Organization: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
                
                # Check for sorting/grouping operations
                elif self._contains_organization_operations(node):
                    pattern = self._create_pattern_analysis(
                        pattern_type="collection_organization",
                        pattern_name=f"Organization Logic: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_suggestion_mechanism_patterns(self, tree: ast.AST, content: str, lines: List[str]) -> List[PatternAnalysis]:
        """Analyze suggestion mechanism patterns in the script"""
        
        patterns = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_name = node.name.lower()
                func_content = ast.unparse(node) if hasattr(ast, 'unparse') else ""
                
                # Check if function is related to suggestions
                if any(keyword in func_name for keyword in self.suggestion_mechanism_keywords):
                    pattern = self._create_pattern_analysis(
                        pattern_type="suggestion_mechanism",
                        pattern_name=f"Suggestion Mechanism: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
                
                # Check for recommendation logic
                elif self._contains_suggestion_operations(node):
                    pattern = self._create_pattern_analysis(
                        pattern_type="suggestion_mechanism",
                        pattern_name=f"Recommendation Logic: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _analyze_other_patterns(self, tree: ast.AST, content: str, lines: List[str]) -> List[PatternAnalysis]:
        """Analyze other interesting patterns in the script"""
        
        patterns = []
        
        # Look for data processing patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_content = ast.unparse(node) if hasattr(ast, 'unparse') else ""
                
                # Check for data validation patterns
                if self._contains_validation_operations(node):
                    pattern = self._create_pattern_analysis(
                        pattern_type="data_validation",
                        pattern_name=f"Data Validation: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
                
                # Check for URL processing patterns
                elif self._contains_url_operations(node):
                    pattern = self._create_pattern_analysis(
                        pattern_type="url_processing",
                        pattern_name=f"URL Processing: {node.name}",
                        node=node,
                        content=func_content,
                        lines=lines
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _create_pattern_analysis(self, pattern_type: str, pattern_name: str, 
                                node: ast.AST, content: str, lines: List[str]) -> PatternAnalysis:
        """Create a pattern analysis from an AST node"""
        
        line_numbers = [node.lineno]
        if hasattr(node, 'end_lineno') and node.end_lineno:
            line_numbers = list(range(node.lineno, node.end_lineno + 1))
        
        # Extract algorithm description
        algorithm_desc = self._describe_algorithm(node, content)
        
        # Determine input/output format
        input_format, output_format = self._analyze_input_output(node)
        
        # Assess complexity
        complexity = self._assess_pattern_complexity(node, content)
        
        # Assess reusability
        reusability = self._assess_reusability(node, content)
        
        # Generate modernization suggestions
        modernization_suggestions = self._generate_pattern_modernization_suggestions(
            pattern_type, node, content
        )
        
        return PatternAnalysis(
            pattern_type=pattern_type,
            pattern_name=pattern_name,
            description=algorithm_desc,
            code_location=f"Line {node.lineno}",
            line_numbers=line_numbers,
            algorithm_description=algorithm_desc,
            input_format=input_format,
            output_format=output_format,
            complexity=complexity,
            reusability=reusability,
            modernization_suggestions=modernization_suggestions
        )
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate simplified cyclomatic complexity"""
        
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        return complexity
    
    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract function calls made within a node"""
        
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        return calls
    
    def _extract_variables_used(self, node: ast.AST) -> Set[str]:
        """Extract variables used within a node"""
        
        variables = set()
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                variables.add(child.id)
        
        return variables
    
    def _detect_function_patterns(self, node: ast.FunctionDef, name: str, docstring: str) -> List[str]:
        """Detect patterns in a function"""
        
        patterns = []
        name_lower = name.lower()
        doc_lower = docstring.lower()
        
        # Check for common patterns
        if any(keyword in name_lower for keyword in self.tag_normalization_keywords):
            patterns.append("tag_normalization")
        
        if any(keyword in name_lower for keyword in self.collection_organization_keywords):
            patterns.append("collection_organization")
        
        if any(keyword in name_lower for keyword in self.suggestion_mechanism_keywords):
            patterns.append("suggestion_mechanism")
        
        # Check algorithm patterns
        for pattern_type, keywords in self.algorithm_patterns.items():
            if any(keyword in name_lower or keyword in doc_lower for keyword in keywords):
                patterns.append(pattern_type)
        
        return patterns
    
    def _detect_class_patterns(self, node: ast.ClassDef, name: str, docstring: str) -> List[str]:
        """Detect patterns in a class"""
        
        patterns = []
        name_lower = name.lower()
        doc_lower = docstring.lower()
        
        # Check for design patterns
        if 'manager' in name_lower or 'handler' in name_lower:
            patterns.append("manager_pattern")
        
        if 'factory' in name_lower:
            patterns.append("factory_pattern")
        
        if 'singleton' in name_lower or 'single' in doc_lower:
            patterns.append("singleton_pattern")
        
        if 'observer' in name_lower or 'listener' in name_lower:
            patterns.append("observer_pattern")
        
        return patterns
    
    def _determine_function_purpose(self, name: str, docstring: Optional[str], patterns: List[str]) -> str:
        """Determine the purpose of a function"""
        
        name_lower = name.lower()
        
        if 'normalize' in name_lower or 'clean' in name_lower:
            return "Data normalization and cleaning"
        elif 'organize' in name_lower or 'sort' in name_lower:
            return "Data organization and sorting"
        elif 'suggest' in name_lower or 'recommend' in name_lower:
            return "Suggestion and recommendation generation"
        elif 'validate' in name_lower or 'check' in name_lower:
            return "Data validation and verification"
        elif 'process' in name_lower or 'handle' in name_lower:
            return "Data processing and handling"
        elif 'parse' in name_lower or 'extract' in name_lower:
            return "Data parsing and extraction"
        else:
            return "General utility function"
    
    def _determine_class_purpose(self, name: str, docstring: Optional[str], 
                                methods: List[FunctionAnalysis], patterns: List[str]) -> str:
        """Determine the purpose of a class"""
        
        name_lower = name.lower()
        
        if 'manager' in name_lower:
            return "Resource management and coordination"
        elif 'handler' in name_lower:
            return "Event handling and processing"
        elif 'analyzer' in name_lower:
            return "Data analysis and processing"
        elif 'importer' in name_lower or 'exporter' in name_lower:
            return "Data import/export operations"
        elif 'validator' in name_lower:
            return "Data validation and verification"
        else:
            return "General purpose class"
    
    def _determine_algorithm_type(self, node: ast.AST, name: str, calls: List[str]) -> str:
        """Determine the type of algorithm used"""
        
        name_lower = name.lower()
        calls_lower = [call.lower() for call in calls]
        
        if any(call in ['sort', 'sorted', 'order'] for call in calls_lower):
            return "sorting_algorithm"
        elif any(call in ['filter', 'select'] for call in calls_lower):
            return "filtering_algorithm"
        elif any(call in ['group', 'groupby'] for call in calls_lower):
            return "grouping_algorithm"
        elif any(call in ['match', 'search', 'find'] for call in calls_lower):
            return "search_algorithm"
        elif 'score' in name_lower or any('score' in call for call in calls_lower):
            return "scoring_algorithm"
        else:
            return "general_algorithm"
    
    def _contains_tag_operations(self, node: ast.AST) -> bool:
        """Check if node contains tag-related operations"""
        
        tag_indicators = ['tag', 'label', 'category', 'normalize', 'clean', 'strip']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Str):
                if any(indicator in child.s.lower() for indicator in tag_indicators):
                    return True
            elif isinstance(child, ast.Name):
                if any(indicator in child.id.lower() for indicator in tag_indicators):
                    return True
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                if child.func.attr.lower() in ['strip', 'lower', 'upper', 'replace', 'normalize']:
                    return True
        
        return False
    
    def _contains_organization_operations(self, node: ast.AST) -> bool:
        """Check if node contains organization-related operations"""
        
        org_indicators = ['sort', 'group', 'organize', 'classify', 'order']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id.lower() in org_indicators:
                    return True
                elif isinstance(child.func, ast.Attribute) and child.func.attr.lower() in org_indicators:
                    return True
        
        return False
    
    def _contains_suggestion_operations(self, node: ast.AST) -> bool:
        """Check if node contains suggestion-related operations"""
        
        suggestion_indicators = ['suggest', 'recommend', 'predict', 'match', 'similar']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if any(indicator in child.id.lower() for indicator in suggestion_indicators):
                    return True
            elif isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                if any(indicator in child.func.id.lower() for indicator in suggestion_indicators):
                    return True
        
        return False
    
    def _contains_validation_operations(self, node: ast.AST) -> bool:
        """Check if node contains validation operations"""
        
        validation_indicators = ['validate', 'check', 'verify', 'assert', 'ensure']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if any(indicator in child.id.lower() for indicator in validation_indicators):
                    return True
            elif isinstance(child, ast.Assert):
                return True
        
        return False
    
    def _contains_url_operations(self, node: ast.AST) -> bool:
        """Check if node contains URL processing operations"""
        
        url_indicators = ['url', 'uri', 'link', 'domain', 'parse', 'extract']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Name):
                if any(indicator in child.id.lower() for indicator in url_indicators):
                    return True
            elif isinstance(child, ast.Str):
                if 'http' in child.s.lower() or 'www' in child.s.lower():
                    return True
        
        return False
    
    def _extract_class_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract class attributes from __init__ method"""
        
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == '__init__':
                for child in ast.walk(item):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                                if target.value.id == 'self':
                                    attributes.append(target.attr)
        
        return attributes
    
    def _describe_algorithm(self, node: ast.AST, content: str) -> str:
        """Generate a description of the algorithm used"""
        
        # Analyze the structure and operations
        operations = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    operations.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    operations.append(child.func.attr)
        
        # Generate description based on operations
        if 'sort' in operations or 'sorted' in operations:
            return "Sorting algorithm that orders data based on specified criteria"
        elif 'filter' in operations:
            return "Filtering algorithm that selects data based on conditions"
        elif 'group' in operations or 'groupby' in operations:
            return "Grouping algorithm that organizes data into categories"
        elif any(op in ['match', 'search', 'find'] for op in operations):
            return "Search algorithm that finds matching patterns or data"
        elif any(op in ['normalize', 'clean', 'strip'] for op in operations):
            return "Normalization algorithm that standardizes data format"
        else:
            return "General processing algorithm for data manipulation"
    
    def _analyze_input_output(self, node: ast.AST) -> Tuple[str, str]:
        """Analyze input and output formats of a function"""
        
        # Simplified analysis based on function signature and body
        input_format = "Mixed data types"
        output_format = "Processed data"
        
        if isinstance(node, ast.FunctionDef):
            # Check parameters for hints
            if node.args.args:
                if any('list' in arg.arg.lower() for arg in node.args.args):
                    input_format = "List/Array data"
                elif any('dict' in arg.arg.lower() for arg in node.args.args):
                    input_format = "Dictionary/Object data"
                elif any('str' in arg.arg.lower() for arg in node.args.args):
                    input_format = "String data"
            
            # Check return statements for output hints
            for child in ast.walk(node):
                if isinstance(child, ast.Return):
                    if isinstance(child.value, ast.List):
                        output_format = "List/Array"
                    elif isinstance(child.value, ast.Dict):
                        output_format = "Dictionary/Object"
                    elif isinstance(child.value, ast.Str):
                        output_format = "String"
                    elif isinstance(child.value, ast.Num):
                        output_format = "Numeric value"
        
        return input_format, output_format
    
    def _assess_pattern_complexity(self, node: ast.AST, content: str) -> str:
        """Assess the complexity of a pattern"""
        
        complexity_score = self._calculate_complexity(node)
        
        if complexity_score <= 3:
            return "simple"
        elif complexity_score <= 7:
            return "moderate"
        else:
            return "complex"
    
    def _assess_reusability(self, node: ast.AST, content: str) -> str:
        """Assess the reusability of a pattern"""
        
        # Check for hardcoded values
        hardcoded_count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.Str, ast.Num)):
                hardcoded_count += 1
        
        # Check for external dependencies
        external_deps = 0
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                external_deps += 1
        
        # Assess based on factors
        if hardcoded_count > 5 or external_deps > 3:
            return "low"
        elif hardcoded_count > 2 or external_deps > 1:
            return "medium"
        else:
            return "high"
    
    def _generate_pattern_modernization_suggestions(self, pattern_type: str, 
                                                   node: ast.AST, content: str) -> List[str]:
        """Generate modernization suggestions for a pattern"""
        
        suggestions = []
        
        # General suggestions based on pattern type
        if pattern_type == "tag_normalization":
            suggestions.extend([
                "Consider using regular expressions for more robust text processing",
                "Implement caching for frequently normalized tags",
                "Add support for Unicode normalization",
                "Consider using a dedicated text processing library"
            ])
        
        elif pattern_type == "collection_organization":
            suggestions.extend([
                "Implement hierarchical organization with tree structures",
                "Add support for multiple organization criteria",
                "Consider using machine learning for automatic categorization",
                "Implement fuzzy matching for similar collections"
            ])
        
        elif pattern_type == "suggestion_mechanism":
            suggestions.extend([
                "Implement machine learning-based recommendations",
                "Add confidence scoring for suggestions",
                "Consider user feedback for improving suggestions",
                "Implement collaborative filtering techniques"
            ])
        
        # Code-specific suggestions
        if self._uses_old_string_methods(node):
            suggestions.append("Replace old string methods with modern alternatives")
        
        if self._has_nested_loops(node):
            suggestions.append("Consider optimizing nested loops for better performance")
        
        if self._lacks_error_handling(node):
            suggestions.append("Add comprehensive error handling and validation")
        
        return suggestions
    
    def _uses_old_string_methods(self, node: ast.AST) -> bool:
        """Check if code uses old string methods"""
        
        old_methods = ['string.join', 'string.split', 'string.strip']
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Attribute):
                method_call = f"{child.func.value.id if isinstance(child.func.value, ast.Name) else ''}.{child.func.attr}"
                if method_call in old_methods:
                    return True
        
        return False
    
    def _has_nested_loops(self, node: ast.AST) -> bool:
        """Check if code has nested loops"""
        
        loop_depth = 0
        max_depth = 0
        
        for child in ast.walk(node):
            if isinstance(child, (ast.For, ast.While)):
                loop_depth += 1
                max_depth = max(max_depth, loop_depth)
            elif isinstance(child, ast.FunctionDef):
                loop_depth = 0
        
        return max_depth > 1
    
    def _lacks_error_handling(self, node: ast.AST) -> bool:
        """Check if code lacks error handling"""
        
        has_try_except = False
        
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                has_try_except = True
                break
        
        return not has_try_except
    
    def _extract_algorithm_insights(self, functions: List[FunctionAnalysis], 
                                   classes: List[ClassAnalysis], content: str) -> Dict[str, Any]:
        """Extract algorithm insights from the analyzed code"""
        
        insights = {
            'algorithm_types': Counter(),
            'complexity_distribution': {'simple': 0, 'moderate': 0, 'complex': 0},
            'common_patterns': [],
            'optimization_opportunities': [],
            'design_principles': []
        }
        
        # Analyze algorithm types
        for func in functions:
            insights['algorithm_types'][func.algorithm_type] += 1
            
            # Categorize by complexity
            if func.complexity_score <= 3:
                insights['complexity_distribution']['simple'] += 1
            elif func.complexity_score <= 7:
                insights['complexity_distribution']['moderate'] += 1
            else:
                insights['complexity_distribution']['complex'] += 1
        
        # Identify common patterns
        all_patterns = []
        for func in functions:
            all_patterns.extend(func.patterns_detected)
        
        pattern_counts = Counter(all_patterns)
        insights['common_patterns'] = [pattern for pattern, count in pattern_counts.most_common(5)]
        
        # Identify optimization opportunities
        high_complexity_funcs = [f for f in functions if f.complexity_score > 7]
        if high_complexity_funcs:
            insights['optimization_opportunities'].append(
                f"Consider refactoring {len(high_complexity_funcs)} high-complexity functions"
            )
        
        # Analyze design principles
        if any('manager' in cls.name.lower() for cls in classes):
            insights['design_principles'].append("Uses manager pattern for resource coordination")
        
        if any('factory' in cls.name.lower() for cls in classes):
            insights['design_principles'].append("Implements factory pattern for object creation")
        
        return insights
    
    def _detect_design_patterns(self, tree: ast.AST, content: str) -> List[str]:
        """Detect design patterns used in the code"""
        
        patterns = []
        
        # Look for singleton pattern
        if 'instance' in content.lower() and '__new__' in content:
            patterns.append("Singleton Pattern")
        
        # Look for factory pattern
        if 'factory' in content.lower() or 'create' in content.lower():
            patterns.append("Factory Pattern")
        
        # Look for observer pattern
        if 'observer' in content.lower() or 'listener' in content.lower():
            patterns.append("Observer Pattern")
        
        # Look for strategy pattern
        if 'strategy' in content.lower() or 'algorithm' in content.lower():
            patterns.append("Strategy Pattern")
        
        return patterns
    
    def _analyze_data_structures(self, tree: ast.AST, content: str) -> List[str]:
        """Analyze data structures used in the code"""
        
        structures = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.List):
                structures.add("List")
            elif isinstance(node, ast.Dict):
                structures.add("Dictionary")
            elif isinstance(node, ast.Set):
                structures.add("Set")
            elif isinstance(node, ast.Tuple):
                structures.add("Tuple")
        
        # Check for imported data structures
        if 'collections' in content:
            if 'defaultdict' in content:
                structures.add("DefaultDict")
            if 'Counter' in content:
                structures.add("Counter")
            if 'deque' in content:
                structures.add("Deque")
        
        return list(structures)
    
    def _generate_modernization_recommendations(self, functions: List[FunctionAnalysis],
                                              classes: List[ClassAnalysis],
                                              tag_patterns: List[PatternAnalysis],
                                              collection_patterns: List[PatternAnalysis],
                                              suggestion_patterns: List[PatternAnalysis]) -> List[str]:
        """Generate overall modernization recommendations"""
        
        recommendations = []
        
        # Function-based recommendations
        high_complexity_count = sum(1 for f in functions if f.complexity_score > 7)
        if high_complexity_count > 0:
            recommendations.append(
                f"Refactor {high_complexity_count} high-complexity functions for better maintainability"
            )
        
        # Pattern-based recommendations
        if tag_patterns:
            recommendations.append("Modernize tag normalization with regex and Unicode support")
        
        if collection_patterns:
            recommendations.append("Implement hierarchical collection organization with ML assistance")
        
        if suggestion_patterns:
            recommendations.append("Enhance suggestion mechanisms with machine learning algorithms")
        
        # General recommendations
        recommendations.extend([
            "Add comprehensive type hints for better code documentation",
            "Implement async/await for I/O operations",
            "Add comprehensive logging and monitoring",
            "Implement configuration management system",
            "Add comprehensive test coverage",
            "Consider using dataclasses for data structures",
            "Implement proper error handling and recovery mechanisms"
        ])
        
        return recommendations
    
    def _identify_reusable_components(self, functions: List[FunctionAnalysis],
                                    classes: List[ClassAnalysis]) -> List[str]:
        """Identify components that can be reused in the new system"""
        
        reusable = []
        
        # Identify utility functions
        utility_functions = [f for f in functions if 'util' in f.name.lower() or 
                           f.algorithm_type in ['string_matching', 'text_processing']]
        
        if utility_functions:
            reusable.extend([f"Utility function: {f.name}" for f in utility_functions])
        
        # Identify data processing functions
        processing_functions = [f for f in functions if 'process' in f.name.lower() or
                              'parse' in f.name.lower()]
        
        if processing_functions:
            reusable.extend([f"Data processing: {f.name}" for f in processing_functions])
        
        # Identify validation functions
        validation_functions = [f for f in functions if 'validate' in f.name.lower() or
                              'check' in f.name.lower()]
        
        if validation_functions:
            reusable.extend([f"Validation logic: {f.name}" for f in validation_functions])
        
        return reusable
    
    def _identify_improvement_opportunities(self, functions: List[FunctionAnalysis],
                                          classes: List[ClassAnalysis],
                                          algorithm_insights: Dict[str, Any]) -> List[str]:
        """Identify opportunities for improvement"""
        
        opportunities = []
        
        # Performance opportunities
        complex_functions = [f for f in functions if f.complexity_score > 10]
        if complex_functions:
            opportunities.append(
                f"Optimize {len(complex_functions)} highly complex functions for better performance"
            )
        
        # Code quality opportunities
        functions_without_docs = [f for f in functions if not f.docstring]
        if functions_without_docs:
            opportunities.append(
                f"Add documentation to {len(functions_without_docs)} undocumented functions"
            )
        
        # Architecture opportunities
        if algorithm_insights['complexity_distribution']['complex'] > 5:
            opportunities.append("Consider breaking down complex algorithms into smaller components")
        
        opportunities.extend([
            "Implement caching for frequently accessed data",
            "Add comprehensive error handling and recovery",
            "Implement proper logging and monitoring",
            "Add performance profiling and optimization",
            "Consider implementing design patterns for better structure"
        ])
        
        return opportunities
    
    def _generate_reference_documentation(self, functions: List[FunctionAnalysis],
                                        classes: List[ClassAnalysis],
                                        tag_patterns: List[PatternAnalysis],
                                        collection_patterns: List[PatternAnalysis],
                                        suggestion_patterns: List[PatternAnalysis]) -> str:
        """Generate comprehensive reference documentation"""
        
        doc_sections = []
        
        # Overview section
        doc_sections.append("# Original Script Analysis Reference\n")
        doc_sections.append("This document provides a comprehensive analysis of the original bookmark management script.\n")
        
        # Functions section
        if functions:
            doc_sections.append("## Functions Analysis\n")
            for func in functions:
                doc_sections.append(f"### {func.name}\n")
                doc_sections.append(f"- **Purpose**: {func.purpose}\n")
                doc_sections.append(f"- **Algorithm Type**: {func.algorithm_type}\n")
                doc_sections.append(f"- **Complexity Score**: {func.complexity_score}\n")
                doc_sections.append(f"- **Parameters**: {', '.join(func.parameters)}\n")
                if func.docstring:
                    doc_sections.append(f"- **Description**: {func.docstring}\n")
                doc_sections.append("")
        
        # Classes section
        if classes:
            doc_sections.append("## Classes Analysis\n")
            for cls in classes:
                doc_sections.append(f"### {cls.name}\n")
                doc_sections.append(f"- **Purpose**: {cls.purpose}\n")
                doc_sections.append(f"- **Methods**: {len(cls.methods)}\n")
                doc_sections.append(f"- **Attributes**: {', '.join(cls.attributes)}\n")
                if cls.docstring:
                    doc_sections.append(f"- **Description**: {cls.docstring}\n")
                doc_sections.append("")
        
        # Patterns section
        if tag_patterns or collection_patterns or suggestion_patterns:
            doc_sections.append("## Identified Patterns\n")
            
            if tag_patterns:
                doc_sections.append("### Tag Normalization Patterns\n")
                for pattern in tag_patterns:
                    doc_sections.append(f"- **{pattern.pattern_name}**: {pattern.description}\n")
            
            if collection_patterns:
                doc_sections.append("### Collection Organization Patterns\n")
                for pattern in collection_patterns:
                    doc_sections.append(f"- **{pattern.pattern_name}**: {pattern.description}\n")
            
            if suggestion_patterns:
                doc_sections.append("### Suggestion Mechanism Patterns\n")
                for pattern in suggestion_patterns:
                    doc_sections.append(f"- **{pattern.pattern_name}**: {pattern.description}\n")
        
        return "\n".join(doc_sections)
    
    def _generate_api_documentation(self, functions: List[FunctionAnalysis],
                                   classes: List[ClassAnalysis]) -> Dict[str, str]:
        """Generate API documentation for functions and classes"""
        
        api_docs = {}
        
        # Document functions
        for func in functions:
            api_docs[f"function_{func.name}"] = f"""
Function: {func.name}
Purpose: {func.purpose}
Parameters: {', '.join(func.parameters)}
Return Type: {func.return_type or 'Unknown'}
Complexity: {func.complexity_score}
Algorithm Type: {func.algorithm_type}
Description: {func.docstring or 'No description available'}
"""
        
        # Document classes
        for cls in classes:
            api_docs[f"class_{cls.name}"] = f"""
Class: {cls.name}
Purpose: {cls.purpose}
Methods: {len(cls.methods)}
Attributes: {', '.join(cls.attributes)}
Inheritance: {', '.join(cls.inheritance)}
Description: {cls.docstring or 'No description available'}
"""
        
        return api_docs
    
    def export_analysis_report(self, analysis_result: ScriptAnalysisResult, 
                              output_path: str, format: str = "json") -> None:
        """Export analysis report to file"""
        
        try:
            output_path = Path(output_path)
            
            if format.lower() == "json":
                # Convert dataclasses to dictionaries for JSON serialization
                report_data = self._convert_analysis_to_dict(analysis_result)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == "markdown":
                # Generate markdown report
                markdown_content = self._generate_markdown_report(analysis_result)
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(markdown_content)
            
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Analysis report exported to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to export analysis report: {e}")
            raise
    
    def _convert_analysis_to_dict(self, analysis: ScriptAnalysisResult) -> Dict[str, Any]:
        """Convert analysis result to dictionary for JSON serialization"""
        
        def convert_dataclass(obj):
            if hasattr(obj, '__dataclass_fields__'):
                return {field: convert_dataclass(getattr(obj, field)) 
                       for field in obj.__dataclass_fields__}
            elif isinstance(obj, list):
                return [convert_dataclass(item) for item in obj]
            elif isinstance(obj, set):
                return list(obj)
            else:
                return obj
        
        return convert_dataclass(analysis)
    
    def _generate_markdown_report(self, analysis: ScriptAnalysisResult) -> str:
        """Generate markdown report from analysis result"""
        
        sections = []
        
        # Header
        sections.append(f"# Original Script Analysis Report\n")
        sections.append(f"**Script**: {analysis.script_path}\n")
        sections.append(f"**Analysis Date**: {analysis.analysis_timestamp}\n")
        sections.append(f"**Total Lines**: {analysis.total_lines}\n")
        sections.append(f"**Functions**: {analysis.total_functions}\n")
        sections.append(f"**Classes**: {analysis.total_classes}\n\n")
        
        # Summary
        sections.append("## Summary\n")
        sections.append(f"This analysis identified {len(analysis.tag_normalization_patterns)} tag normalization patterns, ")
        sections.append(f"{len(analysis.collection_organization_patterns)} collection organization patterns, ")
        sections.append(f"and {len(analysis.suggestion_mechanism_patterns)} suggestion mechanism patterns.\n\n")
        
        # Patterns
        sections.append("## Identified Patterns\n")
        
        if analysis.tag_normalization_patterns:
            sections.append("### Tag Normalization Patterns\n")
            for pattern in analysis.tag_normalization_patterns:
                sections.append(f"- **{pattern.pattern_name}** ({pattern.complexity})\n")
                sections.append(f"  - {pattern.description}\n")
                sections.append(f"  - Location: {pattern.code_location}\n")
                sections.append(f"  - Reusability: {pattern.reusability}\n\n")
        
        if analysis.collection_organization_patterns:
            sections.append("### Collection Organization Patterns\n")
            for pattern in analysis.collection_organization_patterns:
                sections.append(f"- **{pattern.pattern_name}** ({pattern.complexity})\n")
                sections.append(f"  - {pattern.description}\n")
                sections.append(f"  - Location: {pattern.code_location}\n")
                sections.append(f"  - Reusability: {pattern.reusability}\n\n")
        
        if analysis.suggestion_mechanism_patterns:
            sections.append("### Suggestion Mechanism Patterns\n")
            for pattern in analysis.suggestion_mechanism_patterns:
                sections.append(f"- **{pattern.pattern_name}** ({pattern.complexity})\n")
                sections.append(f"  - {pattern.description}\n")
                sections.append(f"  - Location: {pattern.code_location}\n")
                sections.append(f"  - Reusability: {pattern.reusability}\n\n")
        
        # Recommendations
        sections.append("## Modernization Recommendations\n")
        for rec in analysis.modernization_recommendations:
            sections.append(f"- {rec}\n")
        sections.append("\n")
        
        # Reusable Components
        sections.append("## Reusable Components\n")
        for component in analysis.reusable_components:
            sections.append(f"- {component}\n")
        sections.append("\n")
        
        # Reference Documentation
        sections.append("## Reference Documentation\n")
        sections.append(analysis.reference_documentation)
        
        return "".join(sections)