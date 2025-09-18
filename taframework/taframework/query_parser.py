import re
from typing import List, Dict, Any, Optional
from .data_classes import IndicatorConfig
from .indicator_engine import TALibIndicatorEngine
from .enums import ComparisonType
import logging

logger = logging.getLogger(__name__)

class QueryParser:
    COMPARISON_PATTERNS = {
        r'\babove\b': ComparisonType.ABOVE.value,
        r'\bbelow\b': ComparisonType.BELOW.value,
        r'\bcrossed[\s_]?up\b': ComparisonType.CROSSED_UP.value,
        r'\bcrossed[\s_]?down\b': ComparisonType.CROSSED_DOWN.value,
        r'\bequals?\b': ComparisonType.EQUALS.value,
        r'\bgreater[\s_]?than[\s_]?or[\s_]?equal\b': ComparisonType.GREATER_EQUAL.value,
        r'\bless[\s_]?than[\s_]?or[\s_]?equal\b': ComparisonType.LESS_EQUAL.value,
    }

    @classmethod
    def parse_query(cls, query: str) -> List[Dict[str, Any]]:
        operations = []
        for line in query.strip().splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            operation = cls._parse_line(line)
            if operation:
                operations.append(operation)
        return operations

    @classmethod
    def _parse_line(cls, line: str) -> Optional[Dict[str, Any]]:
        line_lower = line.lower()
        comparison = None
        for pattern, comp_type in cls.COMPARISON_PATTERNS.items():
            if re.search(pattern, line_lower):
                comparison = comp_type
                break
        if not comparison:
            logger.warning(f"No valid comparison found in: {line}")
            return None
        parts = re.split(r'\b(?:above|below|crossed[\s_]?(?:up|down)|equals?|greater[\s_]?than[\s_]?or[\s_]?equal|less[\s_]?than[\s_]?or[\s_]?equal)\b',
                        line, flags=re.IGNORECASE)
        if len(parts) < 2:
            logger.warning(f"Malformed query line: {line}")
            return None
        column1 = parts[0].strip()
        column2 = parts[1].strip()
        try:
            column2 = float(column2)
        except ValueError:
            pass
        return {
            'column1': column1,
            'operation': comparison,
            'column2': column2,
            'original_line': line
        }

    @staticmethod
    def extract_indicators(query: str) -> List[IndicatorConfig]:
        indicators = []
        words = re.findall(r'\b[A-Z_]+_\d+\b|\b[A-Z_]+\b', query.upper())
        for word in words:
            if word in ['ABOVE', 'BELOW', 'CROSSED', 'UP', 'DOWN', 'EQUALS']:
                continue
            if '_' in word:
                parts = word.split('_')
                if len(parts) >= 2 and parts[1].isdigit():
                    indicators.append(IndicatorConfig(
                        name=parts[0],
                        period=int(parts[1])
                    ))
            else:
                engine = TALibIndicatorEngine()
                if engine.is_indicator_available(word):
                    indicators.append(IndicatorConfig(name=word))
        return list({(ind.name, ind.period): ind for ind in indicators}.values())
