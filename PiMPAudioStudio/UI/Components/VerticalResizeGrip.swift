import AppKit
import SwiftUI

/// Bottom-edge grip for vertical-only resize of Generate prompt/lyrics fields.
struct VerticalResizeGrip: View {
    @Binding var height: Double
    let range: ClosedRange<Double>

    @State private var dragStartHeight: Double?
    @State private var isHovering = false
    @State private var isDragging = false

    var body: some View {
        ZStack {
            Rectangle()
                .fill(Color.clear)
                .frame(height: 10)
            Capsule()
                .fill((isHovering || isDragging) ? Color.secondary.opacity(0.55) : Color.secondary.opacity(0.28))
                .frame(width: 40, height: 3)
        }
        .frame(maxWidth: .infinity)
        .contentShape(Rectangle())
        .help("Drag to resize height")
        .onHover { hovering in
            isHovering = hovering
            if hovering {
                NSCursor.resizeUpDown.push()
            } else if !isDragging {
                NSCursor.pop()
            }
        }
        .gesture(
            DragGesture(minimumDistance: 1, coordinateSpace: .local)
                .onChanged { value in
                    if dragStartHeight == nil {
                        dragStartHeight = height
                        isDragging = true
                        NSCursor.resizeUpDown.set()
                    }
                    let start = dragStartHeight ?? height
                    let next = start + Double(value.translation.height)
                    height = min(max(next, range.lowerBound), range.upperBound)
                }
                .onEnded { _ in
                    dragStartHeight = nil
                    isDragging = false
                    if !isHovering {
                        NSCursor.pop()
                    }
                }
        )
        .accessibilityLabel("Resize height")
        .accessibilityHint("Drag up or down to change the field height")
        .accessibilityAddTraits(.isButton)
    }
}
