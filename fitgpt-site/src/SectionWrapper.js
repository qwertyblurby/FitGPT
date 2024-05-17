function SectionWrapper({ children }) {
	return (
		<section className="bg-gray-950 text-gray-50 py-12 md:py-24 lg:py-32">
			<div className="container px-4 md:px-6 lg:px-8">
				{children}
			</div>
		</section>
	)
}

export default SectionWrapper;
