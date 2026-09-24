"""Seed a local MongoDB with a small demo dataset.

Creates/updates a `sample_mflix`-style database so the app works out of the
box against `docker-compose`'s MongoDB service — without requiring Atlas.

Usage:
    python scripts/seed_local_mongo.py --uri mongodb://localhost:27017
Add ``--if-empty`` to only insert when the database is empty (handy for
repeated container restarts).

The data deliberately mirrors the structure used by the course notebooks
(``theaters`` and ``movies`` collections) so the same queries shown in the
tutorials work here.
"""

from __future__ import annotations

import argparse

from pymongo import MongoClient

_THEATER_ROWS = [
    (1000, "340 W Market", "Bloomington", "MN", "55425", -93.24565, 44.85466),
    (1001, "1 Grand Ave", "Sacramento", "CA", "95814", -121.4944, 38.5816),
    (1002, "500 Main St", "San Diego", "CA", "92101", -117.1611, 32.7157),
    (1003, "900 Hollywood Blvd", "Los Angeles", "CA", "90028", -118.3267, 34.1016),
    (1004, "300 Congress Ave", "Austin", "TX", "78701", -97.7431, 30.2672),
    (1005, "700 Houston St", "Dallas", "TX", "75202", -96.797, 32.7767),
    (1006, "200 Biscayne Blvd", "Miami", "FL", "33131", -80.1877, 25.7743),
    (1007, "One Times Square", "New York", "NY", "10036", -73.9857, 40.7484),
    (1008, "1500 Sheridan Rd", "Chicago", "IL", "60091", -87.6244, 41.8781),
    (1009, "1621 E Monte Vista Av", "Vacaville", "CA", "95688", -121.96328, 38.367649),
    (1010, "45235 Worth Ave.", "Leonardtown", "MD", "20619", -76.512016, 38.29697),
    (1011, "1000 State St", "San Antonio", "TX", "78205", -98.4936, 29.4241),
]

THEATERS = [
    {
        "theaterId": theater_id,
        "location": {
            "address": {
                "street1": street,
                "city": city,
                "state": state,
                "zipcode": zipcode,
            },
            "geo": {"type": "Point", "coordinates": [lon, lat]},
        },
    }
    for (theater_id, street, city, state, zipcode, lon, lat) in _THEATER_ROWS
]

_MOVIE_ROWS = [
    ("The Godfather", 1972, 9.2, 1700000, ["Crime", "Drama"], ["Marlon Brando", "Al Pacino"]),
    ("The Shawshank Redemption", 1994, 9.3, 2500000, ["Drama"], ["Tim Robbins", "Morgan Freeman"]),
    (
        "The Dark Knight",
        2008,
        9.0,
        2300000,
        ["Action", "Crime", "Drama"],
        ["Christian Bale", "Heath Ledger"],
    ),
    (
        "Pulp Fiction",
        1994,
        8.9,
        1800000,
        ["Crime", "Drama"],
        ["John Travolta", "Samuel L. Jackson"],
    ),
    ("Forrest Gump", 1994, 8.8, 1800000, ["Comedy", "Drama", "Romance"], ["Tom Hanks"]),
    ("Inception", 2010, 8.8, 2000000, ["Action", "Sci-Fi", "Thriller"], ["Leonardo DiCaprio"]),
    ("Interstellar", 2014, 8.6, 1600000, ["Adventure", "Sci-Fi", "Drama"], ["Matthew McConaughey"]),
    ("The Matrix", 1999, 8.7, 1700000, ["Action", "Sci-Fi"], ["Keanu Reeves"]),
    (
        "Toy Story",
        1995,
        8.3,
        900000,
        ["Animation", "Adventure", "Comedy"],
        ["Tom Hanks", "Tim Allen"],
    ),
    ("Fight Club", 1999, 8.8, 1900000, ["Drama"], ["Brad Pitt", "Edward Norton"]),
]

MOVIES = [
    {
        "title": title,
        "year": year,
        "imdb": {"rating": rating, "votes": votes},
        "genres": genres,
        "cast": cast,
    }
    for (title, year, rating, votes, genres, cast) in _MOVIE_ROWS
]


def seed(uri: str, database: str, if_empty: bool, drop: bool) -> None:
    client = MongoClient(uri, serverSelectionTimeoutMS=3000)
    db = client[database]

    if drop:
        db.drop_collection("theaters")
        db.drop_collection("movies")

    if if_empty and db["theaters"].count_documents({}):
        print(f"`{database}` already has data - skipping seed.")
        return

    theaters = db["theaters"].count_documents({})
    movies = db["movies"].count_documents({})
    if theaters == 0:
        db["theaters"].insert_many(THEATERS)
        print(f"Inserted {len(THEATERS)} theaters into `{database}.theaters`.")
    if movies == 0:
        db["movies"].insert_many(MOVIES)
        print(f"Inserted {len(MOVIES)} movies into `{database}.movies`.")

    print("Seeding complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default="mongodb://localhost:27017")
    parser.add_argument("--database", default="sample_mflix")
    parser.add_argument("--if-empty", action="store_true")
    parser.add_argument("--drop", action="store_true")
    args = parser.parse_args()

    seed(args.uri, args.database, args.if_empty, args.drop)


if __name__ == "__main__":
    main()