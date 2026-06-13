
# api-yappin - Outfit Recommendation API

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![Django Version](https://img.shields.io/badge/django-3.2%2B-green.svg)](https://djangoproject.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**Yappify your closet** - An intelligent outfit recommendation system that helps users manage their wardrobe and get personalized outfit suggestions based on occasion and gender preferences.

## 📋 Table of Contents
- [Technical Architecture](#technical-architecture)
- [System Flow Diagram](#system-flow-diagram)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [API Endpoints](#api-endpoints)
- [Database Schema](#database-schema)
- [Usage Examples](#usage-examples)


## 🏗 Technical Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         CLIENT LAYER                            │
│  (Mobile App / Web Frontend / Postman / cURL)                   │
└─────────────────┬───────────────────────────────────────────────┘
                  │ HTTP/HTTPS
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API GATEWAY LAYER                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Authentication│  │   Rate      │  │  Request     │         │
│  │   Middleware  │  │  Limiting   │  │  Logging     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└─────────────────┬───────────────────────────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                    APPLICATION LAYER (Django)                   │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    URL Router (urls.py)                   │  │
│  └────────────┬─────────────────────────────┬────────────────┘  │
│               │                             │                    │
│       ┌───────▼────────┐           ┌────────▼────────┐         │
│       │  User App      │           │  Apparel App    │         │
│       │  ┌──────────┐  │           │  ┌──────────┐   │         │
│       │  │Register  │  │           │  │ Create   │   │         │
│       │  │Login     │  │           │  │ Retrieve │   │         │
│       │  │Profile   │  │           │  │ Update   │   │         │
│       │  └──────────┘  │           │  └──────────┘   │         │
│       └───────┬────────┘           └────────┬────────┘         │
│               │                             │                    │
│               └──────────┬──────────────────┘                    │
│                          │                                        │
│                  ┌───────▼────────┐                              │
│                  │Recommendation  │                              │
│                  │    Engine      │                              │
│                  │ ┌────────────┐ │                              │
│                  │ │Gender Based│ │                              │
│                  │ │Occasion    │ │                              │
│                  │ │Combination │ │                              │
│                  │ │Matcher     │ │                              │
│                  │ └────────────┘ │                              │
│                  └───────┬────────┘                              │
│                          │                                        │
│                  ┌───────▼────────┐                              │
│                  │  Image Handler │                              │
│                  │  (Multimedia)  │                              │
│                  └───────┬────────┘                              │
└─────────────────────────┼─────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  SQLite/     │  │  Media       │  │  Cache       │          │
│  │  PostgreSQL  │  │  Storage     │  │  (Redis)     │          │
│  │  (Database)  │  │  (Images)    │  │  Optional    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## 🔄 System Flow Diagram

```
┌──────────┐
│   USER   │
└─────┬────┘
      │
      ▼
┌─────────────────┐
│ 1. Register/    │
│    Login        │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ 2. Add Apparel  │──────┐
│    Items        │      │
└────────┬────────┘      │
         │               │
         ▼               ▼
┌─────────────────┐ ┌──────────────┐
│ 3. Request      │ │ Image        │
│    Recommendation│ │ Upload       │
└────────┬────────┘ └──────┬───────┘
         │                  │
         ▼                  ▼
┌─────────────────────────────────┐
│    Recommendation Engine        │
│  ┌────────────────────────────┐ │
│  │ Filter by Gender           │ │
│  │ Filter by Occasion         │ │
│  │ Generate Combinations      │ │
│  │ Check Compatibility        │ │
│  └────────────────────────────┘ │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│    4. Return Outfit Suggestions │
│    ┌──────────────────────────┐ │
│    │ Top: Blue Shirt          │ │
│    │ Bottom: Black Jeans      │ │
│    │ Shoes: White Sneakers    │ │
│    │ Accessories: Watch       │ │
│    └──────────────────────────┘ │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│    5. Save Favorite             │
│       Combinations              │
└─────────────────────────────────┘
```

## ✨ Features

| Feature | Status | Description |
|---------|--------|-------------|
| User Registration | ✅ | Create and manage user profiles |
| Apparel Management | ✅ | CRUD operations for clothing items |
| Gender-based Filtering | ✅ | Filter outfits by gender preference |
| Occasion-based Filtering | ✅ | Formal, casual, traditional, party, etc. |
| Outfit Combinations | ✅ | Store and retrieve saved outfits |
| Image Upload | ✅ | Upload and fetch clothing images |
| ML Classification | 🔄 | Future: Auto-categorize from images |
| Weather Integration | 🔄 | Future: Weather-based recommendations |
| Outfit Rating | 🔄 | Future: User rating system |

## 🛠 Tech Stack

```
Backend:
  ├── Python 3.8+
  ├── Django 3.2+
  ├── Django REST Framework
  ├── Django CORS Headers
  └── Pillow (Image Processing)

Database:
  ├── Default: SQLite3
  └── Production: PostgreSQL (recommended)

Storage:
  ├── Local file system (media/)
  └── Future: AWS S3 / Cloud Storage

Authentication:
  └── Django Session Auth / Token Auth
```

## 🚀 Installation

### Prerequisites
```bash
Python 3.8+
pip
virtualenv (recommended)
```

### Step-by-Step Setup

```bash
# 1. Clone the repository
git clone https://github.com/accidentallogger/api-yappin.git
cd api-yappin

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Edit .env with your settings

# 5. Run migrations
python manage.py migrate

# 6. Create superuser
python manage.py createsuperuser

# 7. Run development server
python manage.py runserver

# Server running at http://localhost:8000
```

## 📡 API Endpoints

### Authentication & Users

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| POST | `/api/register/` | Register new user | `{username, email, password, gender}` | User object + token |
| POST | `/api/login/` | User login | `{username, password}` | User object + token |
| GET | `/api/profile/<id>/` | Get user profile | - | User details |
| PUT | `/api/profile/<id>/` | Update profile | `{name, gender, preferences}` | Updated user |
| DELETE | `/api/profile/<id>/` | Delete account | - | Success message |

### Apparel Management

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| POST | `/api/apparel/` | Add clothing item | `{name, category, gender, occasion, image, color}` | Created apparel |
| GET | `/api/apparel/` | Get all apparel (with filters) | Query params: `?gender=male&occasion=formal` | List of apparel |
| GET | `/api/apparel/<id>/` | Get single apparel | - | Apparel details |
| PUT | `/api/apparel/<id>/` | Update apparel | `{name, category, occasion}` | Updated apparel |
| DELETE | `/api/apparel/<id>/` | Delete apparel | - | Success message |
| GET | `/api/apparel/images/` | Get apparel images | - | List of image URLs |
| POST | `/api/apparel/upload-image/` | Upload apparel image | `multipart/form-data: {image}` | Image URL |

### Outfit Recommendations

| Method | Endpoint | Description | Request Body | Response |
|--------|----------|-------------|--------------|----------|
| GET | `/api/recommendations/` | Get outfit suggestions | Query params: `?gender=male&occasion=casual` | List of outfit combinations |
| POST | `/api/recommendations/generate/` | Generate custom recommendation | `{gender, occasion, exclude_items[]}` | Generated outfit |
| GET | `/api/combinations/` | Get saved combinations | - | Saved outfits list |
| POST | `/api/combinations/` | Save outfit combination | `{name, items[], occasion, notes}` | Saved combination |
| GET | `/api/combinations/<id>/` | Get specific combination | - | Combination details |
| DELETE | `/api/combinations/<id>/` | Remove saved combination | - | Success message |

### Sample API Requests

#### Register User
```bash
curl -X POST http://localhost:8000/api/register/ \
  -H "Content-Type: application/json" \
  -d '{
    "username": "fashionista",
    "email": "user@example.com",
    "password": "securepass123",
    "gender": "female"
  }'
```

#### Add Apparel
```bash
curl -X POST http://localhost:8000/api/apparel/ \
  -H "Authorization: Token your_token_here" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Blue Denim Jacket",
    "category": "outerwear",
    "gender": "unisex",
    "occasion": "casual",
    "color": "blue",
    "brand": "Levis"
  }'
```

#### Get Recommendations
```bash
curl -X GET "http://localhost:8000/api/recommendations/?gender=male&occasion=formal" \
  -H "Authorization: Token your_token_here"
```

#### Upload Image with Apparel
```bash
curl -X POST http://localhost:8000/api/apparel/upload-image/ \
  -H "Authorization: Token your_token_here" \
  -F "image=@/path/to/clothing.jpg" \
  -F "apparel_id=123"
```

## 🗄 Database Schema

```sql
-- User Table
CREATE TABLE users (
    id INT PRIMARY KEY,
    username VARCHAR(150),
    email VARCHAR(254),
    gender VARCHAR(20),
    created_at TIMESTAMP
);

-- Apparel Table
CREATE TABLE apparel (
    id INT PRIMARY KEY,
    user_id INT FOREIGN KEY REFERENCES users(id),
    name VARCHAR(200),
    category VARCHAR(50),  -- top, bottom, shoes, accessory
    gender VARCHAR(20),    -- male, female, unisex
    occasion VARCHAR(50),  -- formal, casual, party, traditional
    color VARCHAR(30),
    brand VARCHAR(100),
    image_url VARCHAR(500),
    created_at TIMESTAMP
);

-- Combinations Table
CREATE TABLE combinations (
    id INT PRIMARY KEY,
    user_id INT FOREIGN KEY REFERENCES users(id),
    name VARCHAR(200),
    occasion VARCHAR(50),
    rating INT DEFAULT 0,
    created_at TIMESTAMP
);

-- Combination Items (Many-to-Many)
CREATE TABLE combination_items (
    combination_id INT FOREIGN KEY REFERENCES combinations(id),
    apparel_id INT FOREIGN KEY REFERENCES apparel(id),
    PRIMARY KEY (combination_id, apparel_id)
);
```

## 📊 Entity Relationship Diagram

```
┌─────────────┐        ┌─────────────┐
│    Users    │        │  Apparel    │
├─────────────┤        ├─────────────┤
│ id (PK)     │◄───────│ user_id (FK)│
│ username    │        │ id (PK)     │
│ email       │        │ name        │
│ gender      │        │ category    │
│ password    │        │ gender      │
│ created_at  │        │ occasion    │
└──────┬──────┘        │ color       │
       │               │ image_url   │
       │               └──────┬──────┘
       │                      │
       │              ┌───────┴───────┐
       │              │               │
┌──────▼──────┐        │               │
│Combinations │        │               │
├─────────────┤        │               │
│ id (PK)     │        │               │
│ user_id (FK)│        │               │
│ name        │        │               │
│ occasion    │        │               │
│ rating      │        │               │
│ created_at  │        │               │
└──────┬──────┘        │               │
       │               │               │
       │        ┌──────▼──────────────┐│
       └────────┤ Combination_Items   ││
                ├─────────────────────┤│
                │ combination_id (FK) ││
                │ apparel_id (FK)     │◄┘
                └─────────────────────┘
```

## 🎯 Usage Examples

### Basic Workflow

```python
# 1. Register user
user = register("john_doe", "john@email.com", "male")

# 2. Add clothing items
add_apparel("Black Suit Jacket", category="top", occasion="formal")
add_apparel("Dress Pants", category="bottom", occasion="formal")
add_apparel("Oxford Shoes", category="shoes", occasion="formal")

# 3. Get recommendation for formal event
outfit = recommend(gender="male", occasion="formal")
# Returns: ["Black Suit Jacket", "Dress Pants", "Oxford Shoes"]

# 4. Save favorite combination
save_combination("My Formal Look", outfit_items, occasion="formal")
```

### Filtering Examples

```bash
# Get all casual wear for women
GET /api/apparel/?gender=female&occasion=casual

# Get all blue colored items
GET /api/apparel/?color=blue

# Get party outfits excluding certain items
POST /api/recommendations/generate/
{
    "gender": "female",
    "occasion": "party",
    "exclude_items": [45, 67]
}
```




