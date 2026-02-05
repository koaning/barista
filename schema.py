# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pydantic==2.12.5",
# ]
# ///

"""
Barista Bench Schema Definitions

Shared Pydantic models for coffee order parsing.
Import these in other notebooks with: from schema import Item, Order
"""

from decimal import Decimal
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator, conint


# Valid menu items
ItemName = Literal[
    "Espresso", "Americano", "Drip Coffee", "Latte", "Cappuccino",
    "Flat White", "Mocha", "Caramel Macchiato", "Cold Brew", "Iced Coffee",
    "Frappe (Coffee)", "Frappe (Mocha)", "Strawberry Smoothie", "Chai Latte",
    "Matcha Latte", "Earl Grey Tea", "Green Tea", "Hot Chocolate",
    "Butter Croissant", "Blueberry Muffin", "Bagel", "Avocado Toast",
    "Bacon Gouda Sandwich"
]

# Valid sizes
SizeName = Literal["Short", "Tall", "Grande", "Venti", "Trenta"]

# Valid modifiers
ModifierName = Literal[
    "Oat Milk", "Almond Milk", "Soy Milk", "Coconut Milk", "Breve",
    "Skim Milk", "Vanilla Syrup", "Caramel Syrup", "Hazelnut Syrup",
    "Peppermint Syrup", "Sugar Free Vanilla", "Classic Syrup", "Extra Shot",
    "Whip Cream", "No Whip", "Cold Foam", "Caramel Drizzle", "Extra Hot",
    "Light Ice", "No Ice"
]

# Pricing
BASE_PRICES = {
    "Espresso": 3.00, "Americano": 3.50, "Drip Coffee": 2.50,
    "Latte": 4.50, "Cappuccino": 4.50, "Flat White": 4.75,
    "Mocha": 5.00, "Caramel Macchiato": 5.25, "Cold Brew": 4.25,
    "Iced Coffee": 3.00, "Frappe (Coffee)": 5.50, "Frappe (Mocha)": 5.75,
    "Strawberry Smoothie": 6.00, "Chai Latte": 4.75, "Matcha Latte": 5.25,
    "Earl Grey Tea": 3.00, "Green Tea": 3.00, "Hot Chocolate": 4.00,
    "Butter Croissant": 3.50, "Blueberry Muffin": 3.75, "Bagel": 2.50,
    "Avocado Toast": 7.00, "Bacon Gouda Sandwich": 5.50,
}

SIZE_ADJUST = {
    "Short": -0.50,
    "Tall": 0.00,
    "Grande": 0.50,
    "Venti": 1.00,
    "Trenta": 1.50,
}

MODIFIER_PRICES = {
    "Oat Milk": 0.80, "Almond Milk": 0.60, "Soy Milk": 0.60,
    "Coconut Milk": 0.70, "Breve": 0.80, "Skim Milk": 0.00,
    "Vanilla Syrup": 0.50, "Caramel Syrup": 0.50, "Hazelnut Syrup": 0.50,
    "Peppermint Syrup": 0.50, "Sugar Free Vanilla": 0.50, "Classic Syrup": 0.00,
    "Extra Shot": 1.00, "Whip Cream": 0.50, "No Whip": 0.00,
    "Cold Foam": 1.25, "Caramel Drizzle": 0.50, "Extra Hot": 0.00,
    "Light Ice": 0.00, "No Ice": 0.00,
}


class Item(BaseModel):
    """A single item in a coffee order."""
    name: ItemName
    size: Optional[SizeName] = None
    quantity: conint(ge=1)
    modifiers: list[ModifierName] = Field(default_factory=list)

    @property
    def unit_price(self) -> Decimal:
        price = Decimal(str(BASE_PRICES[self.name]))
        if self.size:
            price += Decimal(str(SIZE_ADJUST[self.size]))
        for m in self.modifiers:
            price += Decimal(str(MODIFIER_PRICES[m]))
        return price

    @property
    def line_total(self) -> Decimal:
        return self.unit_price * self.quantity


class Order(BaseModel):
    """A complete coffee order with items and computed total."""
    items: list[Item]
    total_price: Optional[Decimal] = Field(default=None, json_schema_extra={"hidden": True})

    @model_validator(mode="after")
    def compute_total(self):
        self.total_price = sum(i.line_total for i in self.items)
        return self
