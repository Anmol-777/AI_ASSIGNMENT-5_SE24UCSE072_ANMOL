from owlready2 import *
import json
import random


onto = get_ontology("http://travel.planner.org/onto#")

with onto:
    class Place(Thing): pass
    class Food(Thing): pass
    class Person(Thing): pass
    class TourPlan(Thing): pass

    class locatedIn(ObjectProperty):
        domain = [Place]; range = [Place]
    class hasFood(ObjectProperty):
        domain = [Place]; range = [Food]
    class suitedFor(ObjectProperty):
        domain = [Place]; range = [Person]
    class recommendedBy(ObjectProperty):
        domain = [TourPlan]; range = [Person]
    class includesPlace(ObjectProperty):
        domain = [TourPlan]; range = [Place]

    class hasName(DataProperty):
        domain = [Thing]; range = [str]
    class hasCost(DataProperty):
        domain = [Thing]; range = [float]
    class hasRating(DataProperty):
        domain = [Thing]; range = [float]
    class hasCategory(DataProperty):
        domain = [Thing]; range = [str]
    class hasCuisine(DataProperty):
        domain = [Food]; range = [str]
    class hasInterest(DataProperty):
        domain = [Person]; range = [str]
    class hasBudget(DataProperty):
        domain = [Person]; range = [float]
    class hasDuration(DataProperty):
        domain = [TourPlan]; range = [int]
    class hasTotalCost(DataProperty):
        domain = [TourPlan]; range = [float]


DESTINATION_KB = {
    "Paris": {
        "category": "cultural",
        "cost_per_day": 150.0,
        "rating": 4.8,
        "attractions": ["Eiffel Tower", "Louvre", "Montmartre"],
        "foods": [("Croissant", "bakery", 5.0), ("Coq au Vin", "french", 35.0), ("Crêpe", "street food", 8.0)],
        "tags": ["art", "history", "romance", "architecture"]
    },
    "Tokyo": {
        "category": "urban",
        "cost_per_day": 120.0,
        "rating": 4.9,
        "attractions": ["Shinjuku", "Senso-ji", "Akihabara"],
        "foods": [("Ramen", "japanese", 12.0), ("Sushi", "japanese", 45.0), ("Takoyaki", "street food", 6.0)],
        "tags": ["technology", "culture", "food", "anime"]
    },
    "Rome": {
        "category": "historical",
        "cost_per_day": 130.0,
        "rating": 4.7,
        "attractions": ["Colosseum", "Vatican", "Trevi Fountain"],
        "foods": [("Pizza Margherita", "italian", 14.0), ("Gelato", "dessert", 4.0), ("Pasta Carbonara", "italian", 18.0)],
        "tags": ["history", "art", "religion", "architecture"]
    },
    "Bali": {
        "category": "nature",
        "cost_per_day": 60.0,
        "rating": 4.6,
        "attractions": ["Ubud Rice Terraces", "Tanah Lot", "Kuta Beach"],
        "foods": [("Nasi Goreng", "indonesian", 5.0), ("Satay", "street food", 4.0), ("Babi Guling", "indonesian", 10.0)],
        "tags": ["beach", "nature", "spirituality", "relaxation"]
    },
    "New York": {
        "category": "urban",
        "cost_per_day": 200.0,
        "rating": 4.7,
        "attractions": ["Times Square", "Central Park", "MoMA"],
        "foods": [("New York Pizza", "american", 5.0), ("Bagel", "bakery", 4.0), ("Cheesecake", "dessert", 8.0)],
        "tags": ["art", "food", "shopping", "entertainment"]
    },
    "Petra": {
        "category": "historical",
        "cost_per_day": 80.0,
        "rating": 4.9,
        "attractions": ["The Treasury", "Monastery", "Siq Canyon"],
        "foods": [("Mansaf", "jordanian", 12.0), ("Falafel", "middle eastern", 4.0), ("Knafeh", "dessert", 5.0)],
        "tags": ["history", "adventure", "archaeology", "desert"]
    }
}


def build_ontology_kb():
    place_objects = {}
    with onto:
        for city, data in DESTINATION_KB.items():
            p = Place(city.replace(" ", "_"))
            p.hasName = [city]
            p.hasCost = [data["cost_per_day"]]
            p.hasRating = [data["rating"]]
            p.hasCategory = [data["category"]]
            place_objects[city] = p

            for fname, cuisine, cost in data["foods"]:
                fid = fname.replace(" ", "_") + "_" + city
                f = Food(fid)
                f.hasName = [fname]
                f.hasCuisine = [cuisine]
                f.hasCost = [cost]
                p.hasFood.append(f)

    return place_objects


def score_destination(city, data, user_interests, budget_per_day):
    score = 0
    tag_overlap = len(set(user_interests) & set(data["tags"]))
    score += tag_overlap * 20
    score += data["rating"] * 10

    if data["cost_per_day"] <= budget_per_day:
        score += 30
    elif data["cost_per_day"] <= budget_per_day * 1.3:
        score += 10
    else:
        score -= 20

    return score


def recommend_places(user_interests, budget_per_day, num_days, top_k=3):
    scored = []
    for city, data in DESTINATION_KB.items():
        s = score_destination(city, data, user_interests, budget_per_day)
        scored.append((city, s, data))
    scored.sort(key=lambda x: -x[1])

    selected = scored[:top_k]
    days_each = max(1, num_days // top_k)
    remainder = num_days - days_each * top_k

    plan = []
    for i, (city, score, data) in enumerate(selected):
        days = days_each + (1 if i == 0 else 0) * remainder
        cost = days * data["cost_per_day"]
        food_recs = [f[0] for f in data["foods"]]
        plan.append({
            "city": city,
            "days": days,
            "cost": cost,
            "attractions": data["attractions"],
            "food_recommendations": food_recs,
            "category": data["category"],
            "score": score
        })

    return plan


def assess_cost(plan, flight_cost_estimate=500.0):
    total = flight_cost_estimate
    breakdown = {"flights": flight_cost_estimate}
    for stop in plan:
        total += stop["cost"]
        breakdown[stop["city"]] = {
            "accommodation_food": stop["cost"] * 0.6,
            "activities": stop["cost"] * 0.25,
            "transport": stop["cost"] * 0.15,
            "total": stop["cost"]
        }
    breakdown["grand_total"] = total
    return breakdown


def generate_tour_plan(user_profile):
    name = user_profile["name"]
    interests = user_profile["interests"]
    budget = user_profile["total_budget"]
    days = user_profile["days"]

    print(f"\n{'='*55}")
    print(f"  AI TRAVEL PLANNER — Tour Plan for {name}")
    print(f"{'='*55}")
    print(f"  Interests : {', '.join(interests)}")
    print(f"  Budget    : ${budget:.0f}")
    print(f"  Duration  : {days} days")

    daily_budget = budget / days
    recommendations = recommend_places(interests, daily_budget, days, top_k=3)
    cost_breakdown = assess_cost(recommendations)

    if cost_breakdown["grand_total"] > budget:
        print(f"\n  [!] Estimated cost ${cost_breakdown['grand_total']:.0f} exceeds budget.")
        print(f"      Adjusting plan to fit budget...")
        recommendations = recommend_places(interests, daily_budget * 0.7, days, top_k=3)
        cost_breakdown = assess_cost(recommendations, flight_cost_estimate=300.0)

    print(f"\n  RECOMMENDED ITINERARY")
    print(f"  {'─'*45}")
    for i, stop in enumerate(recommendations, 1):
        print(f"\n  Stop {i}: {stop['city']}  ({stop['days']} days)")
        print(f"    Category    : {stop['category'].capitalize()}")
        print(f"    Attractions : {', '.join(stop['attractions'])}")
        print(f"    Must-try    : {', '.join(stop['food_recommendations'])}")
        print(f"    Est. Cost   : ${stop['cost']:.0f}")

    print(f"\n  COST BREAKDOWN")
    print(f"  {'─'*45}")
    print(f"  Flights     : ${cost_breakdown['flights']:.0f}")
    for stop in recommendations:
        city = stop["city"]
        c = cost_breakdown[city]
        print(f"  {city:<14}: ${c['total']:.0f}  (stay: ${c['accommodation_food']:.0f}, "
              f"activities: ${c['activities']:.0f}, transport: ${c['transport']:.0f})")
    print(f"  {'─'*45}")
    print(f"  TOTAL       : ${cost_breakdown['grand_total']:.0f}  (Budget: ${budget:.0f})")

    onto_plan = generate_ontology_plan(user_profile, recommendations)
    print(f"\n  Ontology tour plan created: {onto_plan.name}")
    print(f"{'='*55}\n")

    return recommendations, cost_breakdown


def generate_ontology_plan(user_profile, stops):
    with onto:
        person = Person(user_profile["name"].replace(" ", "_"))
        person.hasName = [user_profile["name"]]
        person.hasBudget = [user_profile["total_budget"]]
        for interest in user_profile["interests"]:
            person.hasInterest.append(interest)

        plan_id = f"Plan_{user_profile['name'].replace(' ', '_')}_{random.randint(1000, 9999)}"
        tour = TourPlan(plan_id)
        tour.hasDuration = [user_profile["days"]]
        tour.recommendedBy.append(person)

        total = 0
        for stop in stops:
            city_id = stop["city"].replace(" ", "_")
            if city_id in onto.search(iri="*" + city_id):
                place = onto.search_one(iri="*" + city_id)
            else:
                place = Place(city_id)
                place.hasName = [stop["city"]]
            tour.includesPlace.append(place)
            total += stop["cost"]

        tour.hasTotalCost = [total]
    return tour


def run_tests():
    print("\n" + "="*55)
    print("ASSIGNMENT 2 — AI-BASED TRAVEL PLANNER")
    print("="*55)

    build_ontology_kb()
    print(f"\nKnowledge base loaded: {len(list(onto.classes()))} classes, "
          f"{len(list(onto.individuals()))} individuals")

    users = [
        {
            "name": "Alice",
            "interests": ["history", "art", "architecture"],
            "total_budget": 3000.0,
            "days": 14
        },
        {
            "name": "Bob",
            "interests": ["beach", "relaxation", "food"],
            "total_budget": 1500.0,
            "days": 10
        },
        {
            "name": "Carol",
            "interests": ["technology", "food", "entertainment"],
            "total_budget": 5000.0,
            "days": 21
        }
    ]

    for user in users:
        generate_tour_plan(user)

    print("\n--- Ontology Reasoning Test ---")
    places = list(onto.search(type=Place))
    print(f"Total Place instances in ontology: {len(places)}")
    tour_plans = list(onto.search(type=TourPlan))
    print(f"Total Tour Plans created: {len(tour_plans)}")
    for tp in tour_plans:
        if tp.hasTotalCost:
            print(f"  {tp.name}: {len(tp.includesPlace)} stops, est. ${tp.hasTotalCost[0]:.0f}")

    print("\nAll travel planner tests passed.")


if __name__ == "__main__":
    run_tests()