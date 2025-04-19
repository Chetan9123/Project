from flask import Blueprint, jsonify, request, current_app
from flask_login import current_user, login_required
from models import db, PotholePost, Reaction
from datetime import datetime

api = Blueprint('api', __name__)

# ... existing code ...

@api.route('/posts/<int:post_id>/react', methods=['POST'])
@login_required
def react_to_post(post_id):
    """Add or update a reaction to a post"""
    post = PotholePost.query.get_or_404(post_id)
    data = request.get_json()
    
    if not data or 'emoji' not in data:
        return jsonify({'success': False, 'error': 'Emoji is required'}), 400
    
    emoji = data['emoji']
    
    # Check if user already reacted with this emoji
    existing_reaction = Reaction.query.filter_by(
        user_id=current_user.id,
        post_id=post_id,
        emoji=emoji
    ).first()
    
    if existing_reaction:
        # Remove the reaction if it already exists
        db.session.delete(existing_reaction)
        db.session.commit()
        action = 'removed'
    else:
        # Remove any existing reaction by this user
        Reaction.query.filter_by(
            user_id=current_user.id,
            post_id=post_id
        ).delete()
        
        # Add new reaction
        new_reaction = Reaction(
            user_id=current_user.id,
            post_id=post_id,
            emoji=emoji,
            created_at=datetime.utcnow()
        )
        db.session.add(new_reaction)
        db.session.commit()
        action = 'added'
    
    # Get updated reactions for this post
    reactions = Reaction.query.filter_by(post_id=post_id).all()
    reactions_data = [r.to_dict() for r in reactions]
    
    return jsonify({
        'success': True,
        'action': action,
        'reactions': reactions_data
    }) 