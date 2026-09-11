from django.db import models


class StudyResponse(models.Model):
    participant_id = models.CharField(max_length=50)
    design = models.CharField(max_length=20)
    age_range = models.CharField(max_length=20)
    movie_frequency = models.CharField(max_length=20)
    pairwise_observations = models.JSONField(default=list)
    ranking_observations = models.JSONField(default=list)
    elapsed_seconds = models.FloatField()
    completed_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.participant_id} ({self.design})"
