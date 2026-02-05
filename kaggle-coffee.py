# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "diskcache>=5.6.0",
#     "marimo>=0.19.7",
#     "ollama==0.6.1",
#     "pandas>=2.0.0",
#     "pydantic==2.12.5",
#     "pydantic-ai==1.50.0",
#     "python-dotenv==1.2.1",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="columns",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(column=0)
def _(Item, Order):
    order = Order(
        items=[
            Item(
                name="Latte",
                size="Grande",
                quantity=2,
                modifiers=["Oat Milk", "Vanilla Syrup"],
            ),
            Item(
                name="Blueberry Muffin",
                quantity=1,
            ),
        ]
    )
    order.model_dump()
    return


@app.cell
async def _(Order, os):
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    wandb_provider = OpenAIProvider(
        base_url="https://api.inference.wandb.ai/v1",
        api_key=os.environ.get("WANDB_OPENAI_API_KEY")
    )

    ollama_provider = OllamaProvider(base_url='http://localhost:11434/v1')

    model = OpenAIChatModel(
        model_name="openai/gpt-oss-120b",
        provider=wandb_provider,  
    )


    agent = Agent(
        model,
        output_type=Order,
        retries=3,  # Give LLM more attempts if validation fails
        system_prompt="""You are a barista parsing coffee shop orders. Extract the FINAL order after processing any corrections. Only output the final intended order after all corrections are applied.""",
    )

    response = await agent.run('Ill take two Lattes and a Bagel.')
    return Agent, agent, model, response


@app.cell
def _(response):
    response.output.model_dump()
    return


@app.cell
async def _(Agent, model):
    agent_simplify = Agent(
        model,
        output_type=str,
        retries=3,  # Give LLM more attempts if validation fails
        system_prompt="""You are a barista parsing coffee shop orders. Extract the FINAL order after processing any corrections.

    Customers often change their mind mid-sentence using phrases like:
    - "remove that", "scratch that", "nevermind", "cancel that"
    - "wait no", "actually", "hold on"
    - "make that X instead", "change that to"

    When you see these corrections:
    - If they cancel an item entirely, exclude it from the order
    - If they cancel a modifier, exclude that modifier
    - If they change a quantity ("make that two" or "bump that to three"), use the new quantity

    All orders must come from the official menu. 

    ## 1. HOT COFFEES
    - Espresso
    - Americano
    - Drip Coffee
    - Latte
    - Cappuccino
    - Flat White
    - Mocha
    - Caramel Macchiato

    ## 2. COLD / BLENDED
    - Cold Brew
    - Iced Coffee
    - Frappe (Coffee)
    - Frappe (Mocha)
    - Strawberry Smoothie

    ## 3. TEAS & OTHERS
    - Chai Latte
    - Matcha Latte
    - Earl Grey Tea
    - Green Tea
    - Hot Chocolate

    ## 4. FOOD (No modifiers allowed usually)
    - Butter Croissant
    - Blueberry Muffin
    - Bagel
    - Avocado Toast
    - Bacon Gouda Sandwich

    ## 5. SIZES (Drink Price Adjustments)
    - Short (8oz)
    - Tall (12oz)
    - Grande (16oz)
    - Venti (20oz)
    - Trenta (30oz)

    ## 6. MODIFIERS
    ### Milks (Replaces default)
    - Oat Milk
    - Almond Milk
    - Soy Milk
    - Coconut Milk
    - Breve (Half & Half)
    - Skim Milk

    ### Syrups (Add-on)
    - Vanilla Syrup
    - Caramel Syrup
    - Hazelnut Syrup
    - Peppermint Syrup
    - Sugar Free Vanilla
    - Classic Syrup

    ### Toppings / Texture
    - Extra Shot
    - Whip Cream
    - No Whip
    - Cold Foam
    - Caramel Drizzle
    - Extra Hot
    - Light Ice
    - No Ice

    ## 7. LOGIC RULES
    - **Replacement:** If a user asks for a milk (e.g., Oat), it replaces the default.
    - **Cancellation:** If a user says "No Whip" on a drink that has Whip (like Mocha), the price does not change, but "No Whip" is noted.
    - **Double Mods:** Syrups stack (Vanilla + Caramel = +$1.00).
    - **Food:** Food items do not have sizes.

    Only output the final intended order after all corrections are applied.""",
    )

    _response = await agent_simplify.run("Lemme get one tall Strawberry Smoothie include caramel drizzle - remove uh that. Next, I need a like venti drip coffee and extra hot. Oh, and add three trenta chai latte include Sugar Free Vanilla... scratch that one Sugar Free Vanilla. caramel drizzle and make sure no whip. Oh, and add double short mochas.")

    print(_response.output)
    return


@app.cell
async def _(agent):
    _response = await agent.run("""
    **Final Order**

    - Venti Drip Coffee – Extra Hot  
    - 3 × Trenta Chai Latte – Caramel Drizzle, No Whip  
    - 2 × Short Mocha
    """)

    _response.output.model_dump()
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    import marimo as mo
    import os 
    from dotenv import load_dotenv

    load_dotenv(".env");
    return (os,)


@app.cell
def _():
    from decimal import Decimal
    from typing import Literal, Optional
    from pydantic import BaseModel, Field, model_validator, conint

    # Literal types for valid menu items, sizes, and modifiers
    ItemName = Literal[
        "Espresso", "Americano", "Drip Coffee", "Latte", "Cappuccino",
        "Flat White", "Mocha", "Caramel Macchiato", "Cold Brew", "Iced Coffee",
        "Frappe (Coffee)", "Frappe (Mocha)", "Strawberry Smoothie", "Chai Latte",
        "Matcha Latte", "Earl Grey Tea", "Green Tea", "Hot Chocolate",
        "Butter Croissant", "Blueberry Muffin", "Bagel", "Avocado Toast",
        "Bacon Gouda Sandwich"
    ]

    SizeName = Literal["Short", "Tall", "Grande", "Venti", "Trenta"]

    ModifierName = Literal[
        "Oat Milk", "Almond Milk", "Soy Milk", "Coconut Milk", "Breve",
        "Skim Milk", "Vanilla Syrup", "Caramel Syrup", "Hazelnut Syrup",
        "Peppermint Syrup", "Sugar Free Vanilla", "Classic Syrup", "Extra Shot",
        "Whip Cream", "No Whip", "Cold Foam", "Caramel Drizzle", "Extra Hot",
        "Light Ice", "No Ice"
    ]

    BASE_PRICES = {
        "Espresso": 3.00,
        "Americano": 3.50,
        "Drip Coffee": 2.50,
        "Latte": 4.50,
        "Cappuccino": 4.50,
        "Flat White": 4.75,
        "Mocha": 5.00,
        "Caramel Macchiato": 5.25,
        "Cold Brew": 4.25,
        "Iced Coffee": 3.00,
        "Frappe (Coffee)": 5.50,
        "Frappe (Mocha)": 5.75,
        "Strawberry Smoothie": 6.00,
        "Chai Latte": 4.75,
        "Matcha Latte": 5.25,
        "Earl Grey Tea": 3.00,
        "Green Tea": 3.00,
        "Hot Chocolate": 4.00,
        "Butter Croissant": 3.50,
        "Blueberry Muffin": 3.75,
        "Bagel": 2.50,
        "Avocado Toast": 7.00,
        "Bacon Gouda Sandwich": 5.50,
    }

    SIZE_ADJUST = {
        "Short": -0.50,
        "Tall": 0.00,
        "Grande": 0.50,
        "Venti": 1.00,
        "Trenta": 1.50,
    }

    MODIFIER_PRICES = {
        "Oat Milk": 0.80,
        "Almond Milk": 0.60,
        "Soy Milk": 0.60,
        "Coconut Milk": 0.70,
        "Breve": 0.80,
        "Skim Milk": 0.00,
        "Vanilla Syrup": 0.50,
        "Caramel Syrup": 0.50,
        "Hazelnut Syrup": 0.50,
        "Peppermint Syrup": 0.50,
        "Sugar Free Vanilla": 0.50,
        "Classic Syrup": 0.00,
        "Extra Shot": 1.00,
        "Whip Cream": 0.50,
        "No Whip": 0.00,
        "Cold Foam": 1.25,
        "Caramel Drizzle": 0.50,
        "Extra Hot": 0.00,
        "Light Ice": 0.00,
        "No Ice": 0.00,
    }


    class Item(BaseModel):
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
        items: list[Item]
        # Exclude from JSON schema so LLM doesn't try to generate it
        total_price: Optional[Decimal] = Field(default=None, json_schema_extra={"hidden": True})

        @model_validator(mode="after")
        def compute_total(self):
            # Always compute total_price, ignoring any LLM-provided value
            self.total_price = sum(i.line_total for i in self.items)
            return self
    return Item, Order


@app.cell
def _():
    return


@app.cell
def _():
    import json
    import pandas as pd
    from diskcache import Cache

    train_df = pd.read_csv("barista-bench/train.csv")
    sample_df = train_df.head(3)
    cache = Cache(".barista-cache")
    return cache, json, sample_df


@app.cell
async def _(agent, cache, json, sample_df):
    results = []
    for idx, row in sample_df.iterrows():
        order_text = row['order']

        if order_text in cache:
            predicted = cache[order_text]
        else:
            _response = await agent.run(order_text)
            predicted = _response.output.model_dump()
            cache[order_text] = predicted

        expected = json.loads(row['expected_json'])
        match = predicted == expected
        results.append({
            'id': row['id'],
            'order': order_text,
            'predicted': predicted,
            'expected': expected,
            'match': match
        })

    disagreements = [r for r in results if not r['match']]
    accuracy = (len(results) - len(disagreements)) / len(results)
    return accuracy, disagreements, results


@app.cell
def _(accuracy, disagreements):
    print(f"Accuracy: {accuracy:.1%} ({len(disagreements)} disagreements)")
    return


@app.cell
def _(results):
    results
    return


@app.cell
def _(disagreements):
    disagreements[0]
    return


if __name__ == "__main__":
    app.run()
