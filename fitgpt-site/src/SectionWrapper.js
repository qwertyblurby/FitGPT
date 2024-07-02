function SectionWrapper({ id, children }) {
	return (
		<section id={id} className="bg-gray-950 text-gray-50 py-12 md:py-24 lg:py-32">
			<div className="container mx-auto px-8 md:px-12 lg:px-16">
				{children}
			</div>
		</section>
	)
}

export default SectionWrapper;
