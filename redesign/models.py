from django.db import models

class DesignRequest(models.Model):
    original_image = models.ImageField(upload_to='uploads/')
    prompt = models.TextField()
    output_image = models.ImageField(upload_to='outputs/', null=True, blank=True)
    scene_type = models.CharField(max_length=20, default="interior")
    request_hash = models.CharField(max_length=64, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)