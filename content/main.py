import eventlet
eventlet.monkey_patch()

from app import app, socketio, db

application = socketio.WSGIApp(socketio.server, app)

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=True,
        log_output=True,
        use_reloader=True,
        allow_unsafe_werkzeug=True
    )