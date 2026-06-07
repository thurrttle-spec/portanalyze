"""
Crisis scenario models and definitions.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class CrisisScenario:
    """Definition of a geopolitical crisis scenario.
    
    Attributes:
        name: Human-readable name of the crisis
        volatility_multiplier: Factor to multiply volatility (e.g., 1.8 = 80% increase)
        mean_shift: Shift in expected returns (e.g., -0.15 = -15% impact)
        duration_days: Expected crisis duration in trading days
        affected_sectors: List of sectors impacted by the crisis
        description: Detailed description of the scenario
    """
    name: str
    volatility_multiplier: float
    mean_shift: float
    duration_days: int
    affected_sectors: List[str]
    description: str = ""
    
    def __post_init__(self):
        """Validate scenario parameters."""
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be non-empty string")
        
        if not isinstance(self.volatility_multiplier, (int, float)):
            raise TypeError(
                f"volatility_multiplier must be numeric, got {type(self.volatility_multiplier)}"
            )
        
        if self.volatility_multiplier < 1.0:
            raise ValueError(
                f"volatility_multiplier must be >= 1.0, got {self.volatility_multiplier}"
            )
        
        if self.volatility_multiplier > 5.0:
            import warnings
            warnings.warn(
                f"volatility_multiplier {self.volatility_multiplier} is very high. "
                f"Typical crisis scenarios range from 1.2 to 3.0"
            )
        
        if not isinstance(self.mean_shift, (int, float)):
            raise TypeError(f"mean_shift must be numeric, got {type(self.mean_shift)}")
        
        if not -1.0 <= self.mean_shift <= 1.0:
            raise ValueError(
                f"mean_shift must be between -1.0 and 1.0, got {self.mean_shift}"
            )
        
        if not isinstance(self.duration_days, int):
            raise TypeError(f"duration_days must be int, got {type(self.duration_days)}")
        
        if self.duration_days <= 0:
            raise ValueError(
                f"duration_days must be positive, got {self.duration_days}"
            )
        
        if not isinstance(self.affected_sectors, list):
            raise TypeError(
                f"affected_sectors must be list, got {type(self.affected_sectors)}"
            )
    
    @staticmethod
    def us_iran_tension() -> 'CrisisScenario':
        """Create predefined US-Iran geopolitical tension scenario.
        
        Based on historical analysis of Middle East geopolitical events:
        - Oil price volatility increases significantly
        - Energy sector experiences heightened uncertainty
        - Flight to safety impacts equity markets
        
        Returns:
            CrisisScenario configured for US-Iran tensions
        """
        return CrisisScenario(
            name="US-Iran Geopolitical Tension",
            volatility_multiplier=1.8,  # 80% increase in volatility
            mean_shift=-0.15,           # -15% impact on returns
            duration_days=90,           # ~3 months typical duration
            affected_sectors=["Energy", "Defense", "Transportation", "Materials"],
            description=(
                "Scenario modeling heightened US-Iran geopolitical tensions. "
                "Characterized by increased oil price volatility, supply chain disruptions, "
                "and flight-to-safety capital flows. Energy sector particularly impacted "
                "due to potential supply disruptions in Strait of Hormuz."
            )
        )
    
    @staticmethod
    def oil_price_shock() -> 'CrisisScenario':
        """Create oil price shock scenario.
        
        Returns:
            CrisisScenario configured for oil supply disruption
        """
        return CrisisScenario(
            name="Oil Supply Disruption",
            volatility_multiplier=2.2,  # 120% increase in volatility
            mean_shift=-0.20,           # -20% impact on returns
            duration_days=60,           # ~2 months
            affected_sectors=["Energy", "Transportation", "Chemicals"],
            description=(
                "Major oil supply disruption causing price spike. "
                "High volatility in energy stocks, margin pressure on downstream sectors."
            )
        )
    
    @staticmethod
    def moderate_recession() -> 'CrisisScenario':
        """Create moderate economic recession scenario.
        
        Returns:
            CrisisScenario configured for moderate recession
        """
        return CrisisScenario(
            name="Moderate Economic Recession",
            volatility_multiplier=1.5,  # 50% increase in volatility
            mean_shift=-0.25,           # -25% impact on returns
            duration_days=180,          # ~6 months
            affected_sectors=["Energy", "Materials", "Industrials", "Consumer Discretionary"],
            description=(
                "Economic slowdown reducing demand for commodities and energy. "
                "Prolonged impact with gradual recovery expected."
            )
        )
    
    def to_dict(self) -> Dict:
        """Convert scenario to dictionary.
        
        Returns:
            Dictionary representation of the scenario
        """
        return {
            'name': self.name,
            'volatility_multiplier': self.volatility_multiplier,
            'mean_shift': self.mean_shift,
            'duration_days': self.duration_days,
            'affected_sectors': self.affected_sectors,
            'description': self.description
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CrisisScenario(\n"
            f"  name='{self.name}',\n"
            f"  volatility_multiplier={self.volatility_multiplier}x,\n"
            f"  mean_shift={self.mean_shift*100:+.1f}%,\n"
            f"  duration={self.duration_days} days,\n"
            f"  affected_sectors={self.affected_sectors}\n"
            f")"
        )
