"""Local PDF export of saved AI observations; never calls a cloud provider."""
import io
from datetime import datetime, timezone
from urllib.parse import quote, urlsplit
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, KeepTogether


def report_pdf(data, frames, base_url):
    if data['status'] != 'ready' or not data.get('report'):
        raise ValueError('A current completed report is required for PDF export')
    parsed = urlsplit(base_url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Configure a valid dashboard URL for the video link')
    video_url = base_url.rstrip('/') + '/api/incidents/' + quote(data['incident_id'], safe='') + '/play'
    styles = getSampleStyleSheet()
    styles['BodyText'].spaceAfter = 7
    styles['BodyText'].leading = 14
    styles['Heading2'].textColor = colors.HexColor('#163c50')
    def p(value, style='BodyText'):
        return Paragraph(escape(str(value)).replace('\n', '<br/>'), styles[style])
    meta, report = data['metadata'], data['report']
    output = io.BytesIO()
    doc = SimpleDocTemplate(output, pagesize=(595.28, 841.89), rightMargin=42, leftMargin=42,
                            topMargin=42, bottomMargin=42, title='VDMA incident report')
    story = [p('VDMA | Incident report', 'Title'),
             p('Gemini observations - review required', 'Heading2'),
             p('Incident ID: ' + data['incident_id']),
             p(f"Camera: {meta['camera_name']} | Alert: {meta['event_type'].replace('_', ' ')}"),
             p('Detected (UTC): ' + datetime.fromtimestamp(meta['detected_at'], timezone.utc).strftime('%Y-%m-%d %H:%M:%S'))]
    if meta.get('location'):
        story.append(p('Location: ' + str(meta['location'].get('place') or 'Unspecified')))
    usage = data.get('usage') or {}
    story += [p(f"Model: {usage.get('model', 'Gemini')} | Clip: {usage.get('duration_seconds', 'unknown')} seconds | Sampled frames: {usage.get('frames', len(frames))}"),
              p('Assessment: ' + report['assessment'].replace('_', ' '))]
    if frames:
        seconds, jpeg = frames[len(frames)//2]
        width, height = ImageReader(io.BytesIO(jpeg)).getSize()
        scale = min(510/width, 235/height)
        story.append(KeepTogether([Image(io.BytesIO(jpeg), width=width*scale, height=height*scale),
                                  p(f'Evidence frame at {seconds:.1f}s (representative frame, not proof of the alert).')]))
    story += [Paragraph(f'<a href="{escape(video_url, {chr(34): "&quot;"})}" color="#176b94">Open incident video</a>', styles['BodyText']),
              p(video_url), p('Requires access to the VDMA dashboard and a signed-in session. A localhost link works only on the machine running VDMA.'),
              p('Summary', 'Heading2'), p(report['summary'])]
    for title, rows in [('Visible subjects', report['subjects']),
                        ('Timeline (seconds relative to clip)', [f"{row['seconds']:.1f}s: {row['observation']}" for row in report['timeline']]),
                        ('Uncertainties', report['uncertainties'])]:
        if rows:
            story.append(p(title, 'Heading2'))
            story.extend(p(row) for row in rows)
    story += [p('Briefing ' + ('(operator approved)' if data.get('approved') else '(draft)'), 'Heading2'),
              p(report['briefing']), p('AI observations can be incorrect. This report does not confirm identity, intent or guilt.')]
    for observation in data.get('observations', []):
        story += [p('Aftermath observation', 'Heading2'), p(observation['report']['summary'])]
    def footer(canvas, _):
        canvas.setFont('Helvetica', 8)
        canvas.setFillColor(colors.HexColor('#536878'))
        canvas.drawString(42, 24, 'VDMA | Evidence review')
        canvas.drawRightString(553, 24, str(canvas.getPageNumber()))
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return output.getvalue()
