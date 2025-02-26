from django.db import models
from django.contrib.auth.models import User


class FAQ(models.Model):
    question = models.CharField(max_length=200)
    answer = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.question


class ContactMessage(models.Model):
    name = models.CharField(max_length=100)
    email = models.EmailField()
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} - {self.created_at}"


class Prediction(models.Model):
    GENDER_CHOICES = [
        ('M', 'Erkek'),
        ('F', 'Kadın'),
    ]

    # Mevcut alanlar
    user = models.ForeignKey(User, on_delete=models.CASCADE, verbose_name="Kullanıcı")
    image = models.ImageField(upload_to='predictions/', verbose_name="Görsel")
    age = models.IntegerField(verbose_name="Yaş")
    gender = models.CharField(max_length=1, choices=GENDER_CHOICES, verbose_name="Cinsiyet")
    location = models.CharField(max_length=100, verbose_name="Lokasyon")
    smoking_history = models.BooleanField(default=False, verbose_name="Sigara İçme Geçmişi")
    family_history = models.BooleanField(default=False, verbose_name="Aile Geçmişi")

    # Yeni eklenecek alanlar
    smoking_years = models.IntegerField(null=True, blank=True, verbose_name="Sigara İçme Yılları")
    cigarettes_per_day = models.IntegerField(null=True, blank=True, verbose_name="Günlük Sigara Miktarı")
    passive_smoking = models.BooleanField(default=False, verbose_name="Pasif İçicilik")
    quit_smoking_date = models.DateField(null=True, blank=True, verbose_name="Sigara Bırakma Tarihi")

    # Tıbbi Geçmiş
    has_lung_disease = models.BooleanField(default=False, verbose_name="Akciğer Hastalığı Var mı?")
    lung_diseases = models.TextField(blank=True, verbose_name="Akciğer Hastalıkları")
    previous_lung_infections = models.BooleanField(default=False, verbose_name="Önceki Akciğer Enfeksiyonları")
    chronic_cough = models.BooleanField(default=False, verbose_name="Kronik Öksürük")
    cough_duration = models.IntegerField(null=True, blank=True, verbose_name="Öksürük Süresi (Ay)")

    # Risk Faktörleri
    occupational_exposure = models.BooleanField(default=False, verbose_name="Mesleki Maruziyet")
    exposure_details = models.TextField(blank=True, verbose_name="Maruziyet Detayları")
    air_pollution_exposure = models.BooleanField(default=False, verbose_name="Hava Kirliliği Maruziyeti")

    # Semptomlar
    shortness_of_breath = models.BooleanField(default=False, verbose_name="Nefes Darlığı")
    chest_pain = models.BooleanField(default=False, verbose_name="Göğüs Ağrısı")
    weight_loss = models.BooleanField(default=False, verbose_name="Kilo Kaybı")
    fatigue = models.BooleanField(default=False, verbose_name="Yorgunluk")
    sputum_production = models.BooleanField(default=False, verbose_name="Balgam Üretimi")

    # Sonuç alanları
    prediction_result = models.CharField(max_length=100, verbose_name="Tahmin Sonucu")
    confidence_score = models.FloatField(verbose_name="Güven Skoru")
    created_at = models.DateTimeField(auto_now_add=True, verbose_name="Oluşturulma Tarihi")

    def __str__(self):
        return f"{self.user.username} - {self.created_at}"

    class Meta:
        verbose_name = "Tahmin"
        verbose_name_plural = "Tahminler"

class ChatMessage(models.Model):
    user_message = models.TextField()
    bot_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.timestamp}: {self.user_message[:50]}"