# Danial's facial 😏😏🫣

# Enhanced Face Recognition System 🎯

[![Python Version](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This project is an enhanced version of the [face_recognition](https://github.com/ageitgey/face_recognition.git) library, adding several powerful features and improvements for real-world applications.

## ✨ Key Features

### 🎥 Continuous Image Streaming
- Real-time face detection and recognition
- Efficient frame processing
- Optimized performance for video streams

### 🔌 Modular API
- Clean, well-structured API design
- Easy integration with other systems
- Extensible architecture for custom implementations

### 🔄 Emotion Detection Module
- Real-time emotion analysis
- Support for multiple emotion categories
- Accurate sentiment detection

### 💾 SQL Backend Integration
- Robust database management
- Efficient storage and retrieval of face data
- Scalable solution for large datasets

## 🚀 Recent Changes and Updates

- `Danial.customers.add_customer()` now computes the next CustomerID server-side using `SELECT COALESCE(MAX(CustomerID), 0) + 1` and retries on duplicate-key errors a few times. This avoids the race condition that caused IntegrityError when multiple processes attempted to insert the same CustomerID.
- `face_froze.FaceRecognizer.save_new_customer()` was updated to delegate id assignment to the database. It saves the face embedding temporarily, inserts the DB row, then renames the file to `customer_{id}.npy`.

If you want to permanently avoid primary key contention, consider altering the `dbo.customer` table to use an IDENTITY (auto-increment) column for `CustomerID` (backup DB before running migration):

```sql
ALTER TABLE dbo.customer
ADD CustomerID_new INT IDENTITY(1,1);
-- copy data, drop old PK etc.
```

Testing locally:

- Ensure your SQL Server is running and the connection parameters in `Danial.py` are correct.
- Run a small script to add a customer (this will modify your DB):

```bash
python -c "import Danial; c = Danial.customers(); print('next id', c.get_next_customer_id()); c.add_customer(CustomerFaceID='customer_test', Name='Test'); c.close()"
```

Use caution with the command above; it will write to the `ISL` database configured in `Danial.py`.

## 🛠 Installation

```bash
# Clone the repository
git clone https://github.com/anonymousdanial/face-detection.git
cd face-detection

# Install required packages
pip install -r requirements.txt
```

## 💻 Usage

1. Configure your database settings in `Danial.py`
2. Start the face recognition system:
```bash
python face_froze.py
```

## 🏗 Project Structure

```
├── curler.py           # API utilities
├── customer_db.py      # Customer database operations
├── Danial.py          # Core functionality
├── database.py        # Database management
├── emotions.py        # Emotion detection module
├── face_froze.py      # Main application
├── ISL.sql           # Database schema
└── Faces/            # Face encodings storage
    └── face_encodings.npy
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is built upon [face_recognition](https://github.com/ageitgey/face_recognition.git) and inherits its license terms. Additional features and modifications are licensed under the MIT License.

## 👥 Credits

- Original face recognition implementation: [Adam Geitgey](https://github.com/ageitgey)
- Enhanced features and modifications: [anonymousdanial](https://github.com/anonymousdanial)