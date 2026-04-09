from django import forms
from .models import DesignRequest

class DesignForm(forms.ModelForm):
    class Meta:
        model = DesignRequest
        fields = ['original_image', 'prompt']