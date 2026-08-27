import PiMPAudioCore
import SwiftUI

struct CommunityBrowserView: View {
    @StateObject private var service = CommunityService()

    @State private var searchText = ""
    @State private var selectedCategory: ModelCategory?
    @State private var showingImportSheet = false

    /// ModelCategory doesn't conform to CaseIterable — use a hardcoded list.
    private let allCategories: [ModelCategory] = [
        .aceStepDiT, .aceStepVAE, .aceStepLM, .stableAudio,
    ]

    var body: some View {
        VStack(spacing: 0) {
            toolbar
            Divider()
            if let error = service.error {
                errorBanner(error)
            }
            content
            if let lastRefresh = service.lastRefresh {
                footer(lastRefresh)
            }
        }
        .task { await service.refreshManifest() }
        .sheet(isPresented: $showingImportSheet) {
            HuggingFaceImportSheet()
                .environmentObject(service)
                .frame(minWidth: 480, minHeight: 400)
        }
    }

    // MARK: - Toolbar

    private var toolbar: some View {
        HStack(spacing: 8) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
            TextField("Search adapters…", text: $searchText)
                .textFieldStyle(.plain)
            Divider().frame(height: 16)
            Picker("Category", selection: $selectedCategory) {
                Text("All").tag(ModelCategory?.none)
                ForEach(allCategories, id: \.self) { cat in
                    Text(cat.rawValue).tag(ModelCategory?.some(cat))
                }
            }
            .fixedSize()
            Spacer()
            Button {
                showingImportSheet = true
            } label: {
                Label("Import Adapter", systemImage: "plus")
            }
            Button {
                Task { await service.refreshManifest(force: true) }
            } label: {
                if service.isRefreshing {
                    ProgressView().controlSize(.small)
                } else {
                    Label("Refresh", systemImage: "arrow.clockwise")
                }
            }
            .disabled(service.isRefreshing)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        let adapters = filteredAdapters
        if service.isRefreshing, service.manifest == nil {
            loadingState
        } else if adapters.isEmpty {
            emptyState
        } else {
            List(adapters) { adapter in
                AdapterRow(adapter: adapter, service: service)
            }
        }
    }

    private var loadingState: some View {
        VStack(spacing: 12) {
            Spacer()
            ProgressView()
            Text("Loading community adapters…")
                .foregroundStyle(.secondary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
    }

    private var emptyState: some View {
        VStack(spacing: 8) {
            Spacer()
            Image(systemName: "cube.transparent")
                .font(.largeTitle)
                .foregroundStyle(.secondary)
            Text(searchText.isEmpty && selectedCategory == nil
                ? "No Community Adapters"
                : "No Results")
                .font(.headline)
                .foregroundStyle(.secondary)
            Text(searchText.isEmpty && selectedCategory == nil
                ? "Check back later or add one from a HuggingFace URL."
                : "Try a different search or category.")
                .font(.caption)
                .foregroundStyle(.tertiary)
            Spacer()
        }
        .frame(maxWidth: .infinity)
        .padding()
    }

    // MARK: - Error Banner

    private func errorBanner(_ message: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            Text(message)
                .font(.caption)
            Spacer()
            Button("Retry") {
                Task { await service.refreshManifest(force: true) }
            }
            .buttonStyle(.borderless)
            .font(.caption)
            Button {
                service.error = nil
            } label: {
                Image(systemName: "xmark")
            }
            .buttonStyle(.borderless)
            .font(.caption)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 6)
        .background(.yellow.opacity(0.08))
    }

    // MARK: - Footer

    private func footer(_ date: Date) -> some View {
        HStack {
            Spacer()
            Text("Last updated \(date.formatted(.relative(presentation: .named)))")
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 4)
    }

    // MARK: - Filtering

    private var filteredAdapters: [CommunityAdapter] {
        guard let manifest = service.manifest else { return [] }
        return manifest.adapters.filter { adapter in
            let matchesSearch = searchText.isEmpty ||
                adapter.name.localizedCaseInsensitiveContains(searchText) ||
                adapter.author.localizedCaseInsensitiveContains(searchText) ||
                adapter.description.localizedCaseInsensitiveContains(searchText) ||
                adapter.tags.contains { $0.localizedCaseInsensitiveContains(searchText) }
            let matchesCategory = selectedCategory == nil ||
                adapter.baseModelCategory == selectedCategory
            return matchesSearch && matchesCategory
        }
    }
}

// MARK: - Adapter Row

private struct AdapterRow: View {
    let adapter: CommunityAdapter
    @ObservedObject var service: CommunityService

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            // Leading: star badge for featured
            Image(systemName: adapter.featured ? "star.fill" : "star")
                .font(.caption)
                .foregroundStyle(adapter.featured ? .yellow : .clear)
                .frame(width: 14)

            VStack(alignment: .leading, spacing: 4) {
                HStack(spacing: 6) {
                    Text(adapter.name)
                        .font(.body)
                        .fontWeight(.medium)
                    Text("by \(adapter.author)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Text(adapter.description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
                HStack(spacing: 6) {
                    // Rank badge
                    Text("r\(adapter.rank)")
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(.purple.opacity(0.12))
                        .foregroundStyle(.purple)
                        .clipShape(Capsule())
                    // Category badge
                    Text(adapter.baseModelCategory.rawValue)
                        .font(.caption2)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 2)
                        .background(.blue.opacity(0.10))
                        .clipShape(Capsule())
                    // Tags
                    ForEach(adapter.tags.prefix(3), id: \.self) { tag in
                        Text(tag)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 2)
                            .background(.gray.opacity(0.10))
                            .clipShape(Capsule())
                    }
                }
            }
            Spacer()
            downloadButton
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder
    private var downloadButton: some View {
        let isInstalled = service.isInstalled(adapter.id)
        let downloadState = service.downloads[adapter.id]

        if isInstalled {
            Label("Installed", systemImage: "checkmark.circle.fill")
                .font(.caption)
                .foregroundStyle(.green)
        } else if let state = downloadState {
            switch state {
            case let .downloading(progress):
                Button {
                    service.cancelDownload(adapterID: adapter.id)
                } label: {
                    ZStack {
                        CircularProgressView(progress: progress)
                            .frame(width: 24, height: 24)
                        Image(systemName: "xmark")
                            .font(.system(size: 8, weight: .bold))
                    }
                }
                .buttonStyle(.plain)
            case .failed:
                Button("Retry") {
                    service.retryDownload(adapter: adapter)
                }
                .buttonStyle(.borderedProminent)
                .controlSize(.small)
                .tint(.red)
            case .completed:
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            }
        } else {
            Button("Get") {
                service.download(adapter: adapter)
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
        }
    }
}

// MARK: - Circular Progress

private struct CircularProgressView: View {
    let progress: Double

    var body: some View {
        ZStack {
            Circle()
                .stroke(lineWidth: 3)
                .opacity(0.2)
            Circle()
                .trim(from: 0, to: progress)
                .stroke(style: StrokeStyle(lineWidth: 3, lineCap: .round))
                .rotationEffect(.degrees(-90))
                .animation(.linear(duration: 0.2), value: progress)
        }
    }
}
