from django.contrib import admin
from .models import FAQ, ContactMessage, Prediction
from django.utils.html import format_html


@admin.register(FAQ)
class FAQAdmin(admin.ModelAdmin):
    list_display = ("question", "created_at")  # Liste görünümünde gösterilecek sütunlar
    search_fields = ("question", "answer")  # Arama yapılabilecek alanlar
    list_filter = ("created_at",)  # Tarihe göre filtreleme


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ("name", "email", "created_at")
    search_fields = ("name", "email", "message")
    list_filter = ("created_at",)


@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = (
        "user", "age", "gender", "smoking_history", "family_history",
        "prediction_result", "confidence_score", "created_at"
    )
    search_fields = ("user__username", "prediction_result")
    list_filter = (
        "gender", "smoking_history", "family_history",
        "created_at", "shortness_of_breath", "chest_pain", "weight_loss"
    )

    # Admin panelinde resim önizleme ekleyelim
    def image_preview(self, obj):
        if obj.image:
            return format_html(
                '<img src="{}" width="75" height="75" style="border-radius:5px;"/>'.format(obj.image.url)
            )
        return "No Image"

    image_preview.short_description = "Resim Önizleme"

    # Form görünümünü düzenleyelim
    fieldsets = (
        ("Kullanıcı Bilgileri", {"fields": ("user", "age", "gender", "location")}),
        ("Sağlık Geçmişi", {"fields": ("smoking_history", "family_history", "smoking_years",
                                       "cigarettes_per_day", "passive_smoking", "quit_smoking_date",
                                       "has_lung_disease", "lung_diseases", "previous_lung_infections",
                                       "chronic_cough", "cough_duration")}),
        ("Risk Faktörleri", {"fields": ("occupational_exposure", "exposure_details", "air_pollution_exposure")}),
        ("Semptomlar", {"fields": ("shortness_of_breath", "chest_pain", "weight_loss",
                                  "fatigue", "sputum_production")}),
        ("Tahmin Sonucu", {"fields": ("prediction_result", "confidence_score")}),
        ("Yüklenen Görsel", {"fields": ("image_preview", "image")}),
    )

    readonly_fields = ("image_preview",)  # Görsel önizleme alanını salt okunur yap