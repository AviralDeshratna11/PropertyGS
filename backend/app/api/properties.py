"""Property metadata CRUD and Factual Data Layer endpoints."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.domain import Property, PropertyStatus
from app.services.db import get_db

router = APIRouter()


@router.get("/")
async def list_properties(
    city: str = None, status: str = "active",
    min_price: int = None, max_price: int = None,
    limit: int = 20, offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    stmt = select(Property)
    if city:
        stmt = stmt.where(Property.city.ilike(f"%{city}%"))
    if status:
        stmt = stmt.where(Property.status == PropertyStatus(status))
    if min_price:
        stmt = stmt.where(Property.asking_price_usd >= min_price)
    if max_price:
        stmt = stmt.where(Property.asking_price_usd <= max_price)
    stmt = stmt.offset(offset).limit(limit)

    result = await db.execute(stmt)
    props = result.scalars().all()
    return [
        {
            "id": p.id, "title": p.title, "city": p.city,
            "district": p.district, "asking_price_usd": p.asking_price_usd,
            "bedrooms": p.bedrooms, "bathrooms": p.bathrooms,
            "area_sqft": p.area_sqft, "status": p.status.value,
            "lifestyle_match_score": p.lifestyle_match_score,
        }
        for p in props
    ]


@router.get("/{property_id}")
async def get_property(property_id: int, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Property).where(Property.id == property_id))
    p = result.scalar_one_or_none()
    if not p:
        raise HTTPException(404, "Property not found")
    return {
        "id": p.id, "title": p.title, "description": p.description,
        "city": p.city, "district": p.district, "latitude": p.latitude,
        "longitude": p.longitude, "address": p.address,
        "property_type": p.property_type, "bedrooms": p.bedrooms,
        "bathrooms": p.bathrooms, "area_sqft": p.area_sqft,
        "asking_price_usd": p.asking_price_usd,
        "sunlight_score": p.sunlight_score, "noise_score": p.noise_score,
        "walk_score": p.walk_score, "commute_minutes": p.commute_minutes,
        "lifestyle_match_score": p.lifestyle_match_score,
        "rera_number": p.rera_number, "status": p.status.value,
        "is_verified": p.is_verified,
    }


@router.post("/seed")
async def seed_properties(db: AsyncSession = Depends(get_db)):
    """Seed database with sample Dubai and NYC properties for development."""
    samples = [
        Property(
            external_id="DXB-001", title="Palm Jumeirah 4BR Villa with Private Beach",
            description="Luxury waterfront villa with panoramic sea views",
            city="Dubai", district="Palm Jumeirah", latitude=25.1124, longitude=55.1390,
            address="Frond N, Palm Jumeirah, Dubai",
            property_type="villa", bedrooms=4, bathrooms=5, area_sqft=5200,
            floor_number=0, year_built=2020, parking_spaces=3,
            asking_price_usd=3_500_000, price_per_sqft=673, currency="USD",
            sunlight_score=88, noise_score=78, walk_score=42, transit_score=35,
            rera_number="RERA-2024-78421",
        ),
        Property(
            external_id="DXB-002", title="Downtown Dubai 2BR with Burj Khalifa View",
            description="Modern apartment in Boulevard Point with iconic city views",
            city="Dubai", district="Downtown Dubai", latitude=25.1972, longitude=55.2744,
            address="Boulevard Point, Downtown Dubai",
            property_type="apartment", bedrooms=2, bathrooms=3, area_sqft=1450,
            floor_number=38, year_built=2022, parking_spaces=1,
            asking_price_usd=950_000, price_per_sqft=655, currency="USD",
            sunlight_score=72, noise_score=55, walk_score=85, transit_score=78,
            rera_number="RERA-2024-91002",
        ),
        Property(
            external_id="DXB-003", title="Arabian Ranches III 5BR Family Villa",
            description="Spacious family home in gated community with landscaped gardens",
            city="Dubai", district="Arabian Ranches III", latitude=25.0587, longitude=55.2518,
            address="Arabian Ranches III, Dubai",
            property_type="villa", bedrooms=5, bathrooms=6, area_sqft=6800,
            floor_number=0, year_built=2024, parking_spaces=4,
            asking_price_usd=2_200_000, price_per_sqft=324, currency="USD",
            sunlight_score=92, noise_score=90, walk_score=28, transit_score=20,
            rera_number="RERA-2025-10334",
        ),
        Property(
            external_id="NYC-001", title="Tribeca Loft with Hudson River Views",
            description="Sun-drenched industrial conversion loft in prime Tribeca",
            city="New York", district="Tribeca", latitude=40.7163, longitude=-74.0086,
            address="155 Franklin St, New York, NY 10013",
            property_type="apartment", bedrooms=3, bathrooms=2, area_sqft=2800,
            floor_number=6, year_built=1920, parking_spaces=0,
            asking_price_usd=4_200_000, price_per_sqft=1500, currency="USD",
            sunlight_score=65, noise_score=45, walk_score=98, transit_score=95,
        ),
        Property(
            external_id="NYC-002", title="UES Classic 6 with Central Park Proximity",
            description="Pre-war elegance with modern renovations near the Park",
            city="New York", district="Upper East Side", latitude=40.7736, longitude=-73.9566,
            address="1060 Park Ave, New York, NY 10128",
            property_type="apartment", bedrooms=3, bathrooms=3, area_sqft=2200,
            floor_number=12, year_built=1928, parking_spaces=0,
            asking_price_usd=3_100_000, price_per_sqft=1409, currency="USD",
            sunlight_score=58, noise_score=60, walk_score=95, transit_score=90,
        ),
    ]

    for p in samples:
        db.add(p)
    await db.commit()
    return {"seeded": len(samples), "message": "Sample properties loaded"}
